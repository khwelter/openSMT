# Architectural Re-Evaluation: Placement Head Feature

**Date:** April 29, 2026  
**Context:** Current system is stable with command tracking, job system, and conflict policy. Incoming placement head feature introduces multi-nozzle offset geometry and valve control.

---

## 1. Requirements Summary

### Hardware Model
- **Placement head** contains 4+ nozzles (configurable per system)
- **Each nozzle** has:
  - One **vacuum valve** (always present) — can apply suction
  - Optional **pressurized-air valve** — for gentle release
  - **XY offset** (dx, dy) relative to **camera position**
- **Camera is THE machine reference** (0, 0 = home position)
- Machine position is stored in PositionStore

### UI Requirements
- Per-nozzle card with:
  - Status display (position, valve states)
  - **Button 1:** "→ Align to Camera" — move nozzle to camera's current position
  - **Button 2:** "← Move Camera Here" — move camera (machine) to nozzle's current position

---

## 2. Current Architecture Readiness Assessment

### ✅ What's Good (Ready for Nozzles)
1. **PositionStore** — single source of truth for machine position (X, Y, Z*, R*)
2. **HardwareDriver** — abstracts motion commands (jog, home, etc.)
3. **CommandRunner** — job tracking, cancellation, conflict policy (auto-cancel per domain)
4. **Domain-aware submission** — coord moves already route through conflict policy
5. **Serial boards** — can carry additional IO (valve control via IO pins/Marlin)

### ⚠️ Architectural Gaps for Placement Head
1. **No valve state store** — need new state container for vacuum/air per nozzle
2. **No nozzle geometry model** — offsets hardcoded nowhere; only Z-axis per nozzle defined
3. **Motion calculations don't account for offsets** — jog commands assume camera = reference directly
4. **No valve IO abstraction** — unclear how vacuum/air pins map to Marlin boards
5. **CameraVisionModule growing large** — 1850+ lines; adding nozzle sync buttons + valve control will exceed 2000
6. **No per-nozzle "active domain"** — conflict policy is coord/head; nozzle movements need their own conflict handling

---

## 3. Recommended Improvements (Priority Order)

### **PRIORITY 1: Config Schema — Nozzle Geometry**

#### Current Issue
```json
"camera": {
  "nozzles": [
    { "name": "N1", "z_axis": "Z1", "min_z": -50.0, "max_z": 0.0 }
  ]
}
```
This is incomplete — missing XY offsets and valve configuration.

#### Recommendation
```json
"camera": {
  "nozzles": [
    {
      "name": "N1",
      "z_axis": "Z1",
      "min_z": -50.0,
      "max_z": 0.0,
      "offset_x": 10.5,
      "offset_y": -8.3,
      "vacuum_valve": {
        "board": "AB",
        "io_type": "gpio|pwm|relay",
        "pin": 5
      },
      "air_valve": {
        "board": "AB",
        "io_type": "gpio|pwm|relay",
        "pin": 6
      }
    },
    {
      "name": "N2",
      "z_axis": "Z2",
      "min_z": -50.0,
      "max_z": 0.0,
      "offset_x": -10.5,
      "offset_y": -8.3,
      "vacuum_valve": { "board": "AB", "io_type": "gpio", "pin": 7 }
    }
  ]
}
```

**Impact:**
- Define nozzle geometry centrally (no magic numbers)
- Enable multi-configuration support (4, 6, 8 nozzles per machine type)
- Make valve hardware explicit (board/pin/type)

---

### **PRIORITY 2: Valve State Store**

#### Current Issue
PositionStore only tracks axis positions. Valve states (vacuum on/off, air on/off per nozzle) have nowhere to live.

#### Recommendation: Create `ValveStore` (new module)

**File:** `src/opensmt/store/valve_store.py`

```python
from dataclasses import dataclass, field

@dataclass
class NozzleValveState:
    nozzle_name: str
    vacuum_on: bool = False
    air_on: bool = False

class ValveStore:
    """Single source of truth for all nozzle valve states."""
    
    def __init__(self, nozzle_names: list[str]) -> None:
        self._states = {name: NozzleValveState(name) for name in nozzle_names}
        self._callbacks: list = []
    
    def subscribe(self, callback) -> None: ...
    def unsubscribe(self, callback) -> None: ...
    
    async def set_vacuum(self, nozzle_name: str, on: bool) -> None:
        """Update vacuum valve state."""
        if nozzle_name not in self._states:
            return
        self._states[nozzle_name].vacuum_on = on
        await self._notify()
    
    async def set_air(self, nozzle_name: str, on: bool) -> None:
        """Update air valve state."""
        if nozzle_name not in self._states:
            return
        self._states[nozzle_name].air_on = on
        await self._notify()
    
    def get(self, nozzle_name: str) -> NozzleValveState | None: ...
    def all(self) -> dict[str, NozzleValveState]: ...
```

**Impact:**
- Valve state is observable (subscribers notified)
- Frontend can poll valve state via API
- Dashboard reflects actual hardware state

---

### **PRIORITY 3: Nozzle Geometry Model**

#### Current Issue
No Python dataclass to represent nozzle configuration and offsets. Hard to calculate movements.

#### Recommendation: Create `NozzleConfig` (extend store package)

**File:** `src/opensmt/store/nozzle_config.py`

```python
from dataclasses import dataclass

@dataclass
class ValveConfig:
    board: str          # e.g., "AB"
    io_type: str        # "gpio", "pwm", "relay"
    pin: int

@dataclass
class NozzleConfig:
    name: str
    z_axis: str         # e.g., "Z1"
    min_z: float
    max_z: float
    offset_x: float     # Relative to camera position
    offset_y: float
    vacuum_valve: ValveConfig
    air_valve: ValveConfig | None = None

class NozzleConfigStore:
    """Immutable nozzle geometry, loaded from config at startup."""
    
    def __init__(self, nozzles: list[NozzleConfig]) -> None:
        self._nozzles = {n.name: n for n in nozzles}
    
    def get(self, nozzle_name: str) -> NozzleConfig | None: ...
    def all(self) -> dict[str, NozzleConfig]: ...
```

**Impact:**
- Type-safe nozzle geometry
- Easy to calculate nozzle absolute position from machine position + offset
- Single place to query nozzle configuration

---

### **PRIORITY 4: Motion Math for Nozzle Movements**

#### Current Issue
When user clicks "Move Nozzle to Camera Position", the system doesn't know how to calculate the target machine position accounting for the nozzle offset.

#### Recommendation: Add helper methods to HardwareDriver

**Location:** `src/opensmt/hardware/driver.py`

```python
async def jog_nozzle_to_camera_position(
    self,
    nozzle_name: str,
    nozzle_config: NozzleConfig,
    current_camera_position: tuple[float, float],
    velocity: float = None
) -> bool:
    """
    Move nozzle to align with camera's current position.
    
    Since: nozzle_position = camera_position + offset
    To achieve nozzle at (cx, cy), move machine to (cx - offset_x, cy - offset_y)
    """
    machine_target_x = current_camera_position[0] - nozzle_config.offset_x
    machine_target_y = current_camera_position[1] - nozzle_config.offset_y
    
    return await self.jog_xy(machine_target_x, machine_target_y, velocity)

async def jog_camera_to_nozzle_position(
    self,
    nozzle_name: str,
    nozzle_config: NozzleConfig,
    current_nozzle_position: tuple[float, float],
    velocity: float = None
) -> bool:
    """
    Move camera (machine) to align with nozzle's current position.
    
    Since: nozzle_position = camera_position + offset
    To achieve camera at nozzle position, move machine to the nozzle position.
    """
    return await self.jog_xy(
        current_nozzle_position[0],
        current_nozzle_position[1],
        velocity
    )
```

**Impact:**
- Offset math is encapsulated in one place
- Motion commands understand nozzle geometry
- Easy to test offset calculations

---

### **PRIORITY 5: Valve Control Endpoints & HardwareDriver**

#### Current Issue
No API or hardware abstraction for valve control. How do we toggle vacuum?

#### Recommendation: Extend HardwareDriver with valve methods

**Location:** `src/opensmt/hardware/driver.py`

```python
async def set_nozzle_vacuum(self, nozzle_name: str, on: bool) -> bool:
    """
    Activate/deactivate vacuum valve on nozzle.
    
    Looks up nozzle config, finds valve pin/board, issues IO command to board.
    """
    # 1. Look up nozzle config
    # 2. Find which board and pin
    # 3. Send IO command to SerialBoard
    # 4. Update ValveStore
    # 5. Return success/failure
    pass

async def set_nozzle_air(self, nozzle_name: str, on: bool) -> bool:
    """Activate/deactivate air valve on nozzle."""
    pass
```

**Impact:**
- Valve control is hardware-agnostic (abstracted from app layer)
- SerialBoard handles actual IO
- ValveStore stays in sync with hardware

---

### **PRIORITY 6: API Endpoints for Nozzle Control**

#### Current Issue
CameraVisionModule (camera_vision.py) has no endpoints for:
- Toggle valve per nozzle
- Move nozzle to camera
- Move camera to nozzle

#### Recommendation: Add these endpoints to CameraVisionModule

```python
# In CameraVisionModule

@routes.post("/api/nozzle/{nozzle_name}/vacuum")
async def nozzle_vacuum(request: web.Request) -> web.Response:
    """Toggle vacuum valve: POST /api/nozzle/N1/vacuum?on=true"""
    nozzle_name = request.match_info["nozzle_name"]
    on = request.rel_url.query.get("on", "false").lower() == "true"
    
    job_id = await self._submit_domain_command(
        "nozzle", f"vacuum_{nozzle_name}", 
        lambda: self.driver.set_nozzle_vacuum(nozzle_name, on)
    )
    return web.json_response({"job_id": job_id})

@routes.post("/api/nozzle/{nozzle_name}/air")
async def nozzle_air(request: web.Request) -> web.Response:
    """Toggle air valve: POST /api/nozzle/N1/air?on=true"""
    # Similar structure

@routes.post("/api/nozzle/{nozzle_name}/move-to-camera")
async def nozzle_to_camera(request: web.Request) -> web.Response:
    """Move nozzle to align with camera position."""
    nozzle_name = request.match_info["nozzle_name"]
    
    cam_x = self.position_store.get("X")
    cam_y = self.position_store.get("Y")
    nozzle_cfg = self.nozzle_config.get(nozzle_name)
    
    job_id = await self._submit_domain_command(
        "nozzle", f"move_nozzle_to_camera_{nozzle_name}",
        lambda: self.driver.jog_nozzle_to_camera_position(
            nozzle_name, nozzle_cfg, (cam_x, cam_y)
        )
    )
    return web.json_response({"job_id": job_id})

@routes.post("/api/nozzle/{nozzle_name}/move-camera-here")
async def camera_to_nozzle(request: web.Request) -> web.Response:
    """Move camera (machine) to align with nozzle's position."""
    nozzle_name = request.match_info["nozzle_name"]
    
    nozzle_cfg = self.nozzle_config.get(nozzle_name)
    nozzle_x = self.position_store.get("X") + nozzle_cfg.offset_x
    nozzle_y = self.position_store.get("Y") + nozzle_cfg.offset_y
    
    job_id = await self._submit_domain_command(
        "nozzle", f"move_camera_to_nozzle_{nozzle_name}",
        lambda: self.driver.jog_camera_to_nozzle_position(
            nozzle_name, nozzle_cfg, (nozzle_x, nozzle_y)
        )
    )
    return web.json_response({"job_id": job_id})
```

**Impact:**
- All nozzle operations routed through CommandRunner job system
- Valve changes and movements are observable (job tracking)
- Conflict policy applies to nozzle domain (auto-cancel previous active)

---

### **PRIORITY 7: Update Domain Conflict Policy**

#### Current Issue
Conflict policy only handles "coord" and "head" domains. Nozzle movements should have their own domain per nozzle, OR one shared "nozzle" domain.

#### Recommendation: Single "nozzle" domain for all nozzle operations

**Rationale:**
- Nozzle movements (sync buttons) should not interfere with each other in rapid clicks
- Valve toggles + movements on same nozzle should serialize
- Simpler than per-nozzle domain tracking

**Implementation:**
In `camera_vision.py`, `_submit_domain_command()` already handles domain-based auto-cancel:
```python
if domain == "nozzle":
    # Only one nozzle operation at a time
    # Auto-cancel any previous nozzle job
    await self._cancel_active_job_for_domain("nozzle")
```

**Impact:**
- User can click buttons rapidly; only last click executes
- Previous operation is canceled gracefully
- UI shows "Previous command canceled" feedback

---

### **PRIORITY 8: Frontend State & Job Tracking**

#### Current Issue
Frontend needs to display:
- Per-nozzle position (accounting for offset)
- Valve states (vacuum on/off, air on/off)
- Active nozzle job status

#### Recommendation: Extend JavaScript API state

In `camera_vision.py`, update `/api/status` endpoint to include:
```json
{
  "position": { "X": 100, "Y": 200, "Z1": -25, ... },
  "nozzles": [
    {
      "name": "N1",
      "z_axis": "Z1",
      "z_position": -25,
      "x_absolute": 110.5,
      "y_absolute": 191.7,
      "offset_x": 10.5,
      "offset_y": -8.3,
      "vacuum_on": false,
      "air_on": false,
      "active_job_id": null
    }
  ],
  "active_coord_job_id": null,
  "active_head_job_id": null,
  "active_nozzle_job_id": null
}
```

Frontend JavaScript:
```javascript
function getNozzleAbsolutePosition(nozzle) {
  const camX = state.position.X;
  const camY = state.position.Y;
  return {
    x: camX + nozzle.offset_x,
    y: camY + nozzle.offset_y
  };
}

async function moveNozzleToCamera(nozzleName) {
  return submitTrackedCommand(
    `POST /api/nozzle/${nozzleName}/move-to-camera`,
    "nozzle"  // domain
  );
}
```

**Impact:**
- UI shows accurate nozzle positions
- Frontend polls valve state from API
- Job tracking works for nozzle operations

---

### **PRIORITY 9: Module Organization (Design Decision)**

#### Current Issue
CameraVisionModule is 1850+ lines and growing. Adding nozzle support + valve control will push it beyond 2500 lines.

#### Recommendation: **DEFER module splitting, but prepare for it**

**Immediate action (next 2 weeks):**
- Keep all nozzle code in CameraVisionModule
- Use clear section comments to organize:
  ```python
  # =========================================================================
  # NOZZLE VALVE CONTROL
  # =========================================================================
  @routes.post("/api/nozzle/{nozzle_name}/vacuum")
  async def ...: ...
  ```

**Post-feature (planned future refactor):**
Split into:
1. **CameraStreamController** — WebSocket, camera frame broadcasting
2. **DashboardApiController** — Status, coordinate, light, nozzle endpoints
3. **MotionCommandHandler** — Wires job submission, conflict policy
4. **AssetRenderer** — HTML templates, icon serving

**Why defer?**
- Splitting mid-feature risks incomplete implementations
- Nozzle feature scope not fully known yet
- Better to have one complete, then split cleanly

**Impact:**
- CameraVisionModule remains single, manageable file for now
- Clear comments delineate nozzle section
- Next refactor is straightforward (copy section → new controller)

---

### **PRIORITY 10: IO Abstraction for Valve Control**

#### Current Issue
How do vacuum/air valves connect to Marlin boards? GPIO? PWM? Relay?

#### Recommendation: Extend SerialBoard with IO abstraction

**File:** `src/opensmt/hardware/board.py`

```python
class SerialBoard:
    async def set_io(self, pin: int, io_type: str, value: bool | int) -> bool:
        """
        Set IO pin to a value.
        
        io_type: "gpio" → M42 P<pin> S<0|1>
                 "pwm"  → M42 P<pin> S<0-255> (PWM value)
                 "relay" → M42 P<pin> S<0|1>
        """
        if io_type == "gpio" or io_type == "relay":
            cmd = f"M42 P{pin} S{1 if value else 0}"
        elif io_type == "pwm":
            cmd = f"M42 P{pin} S{int(value)}"
        else:
            return False
        
        return await self.command(cmd)
```

**HardwareDriver integration:**
```python
async def set_nozzle_vacuum(self, nozzle_name: str, on: bool) -> bool:
    cfg = self.nozzle_config.get(nozzle_name)
    if not cfg or not cfg.vacuum_valve:
        return False
    
    board = self.boards[cfg.vacuum_valve.board]
    return await board.set_io(
        cfg.vacuum_valve.pin,
        cfg.vacuum_valve.io_type,
        on
    )
```

**Impact:**
- IO control is generic (works with GPIO, PWM, relay)
- Config defines valve hardware detail
- Hardware abstraction is complete

---

## 4. Data Flow Diagram

### Scenario: "Move Nozzle N1 to Camera Position"

```
User clicks button "→ Align to Camera" on N1 card
         ↓
Frontend JavaScript:
  moveNozzleToCamera("N1") → submitTrackedCommand(POST /api/nozzle/N1/move-to-camera, "nozzle")
         ↓
CameraVisionModule:
  POST /api/nozzle/N1/move-to-camera
    ↓
  _submit_domain_command("nozzle", "move_nozzle_to_camera_N1", ...)
    ↓
  [Auto-cancel previous active nozzle job if any]
    ↓
  CommandRunner.submit(job)
    ↓
  HardwareDriver.jog_nozzle_to_camera_position(N1, nozzle_cfg, (cam_x, cam_y))
    ↓
  Calculate machine target:
    target_x = cam_x - nozzle_cfg.offset_x
    target_y = cam_y - nozzle_cfg.offset_y
    ↓
  HardwareDriver.jog_xy(target_x, target_y)
    ↓
  SerialBoard("XY").jog_xy(...) → sends GCODE to Marlin
    ↓
  Machine moves
    ↓
  SerialBoard reads back position → PositionStore.update("X", ...), .update("Y", ...)
    ↓
  Frontend polls /api/status, sees X/Y changed
    ↓
  UI updates nozzle N1 absolute position display
    ↓
Frontend polls /api/jobs/{job_id}
    ↓
  When status = "succeeded", show success feedback
```

---

## 5. Implementation Roadmap

### Phase 1: Infrastructure (Week 1)
- [ ] Create `ValveStore` class (valve state tracking)
- [ ] Create `NozzleConfig` classes + `NozzleConfigStore`
- [ ] Update `system.json` schema with nozzle offsets, valve config
- [ ] Add IO abstraction to SerialBoard (`set_io()` method)

### Phase 2: Hardware Drivers (Week 1)
- [ ] Add nozzle motion helpers to HardwareDriver (jog_nozzle_to_camera_position, etc.)
- [ ] Add valve control helpers to HardwareDriver (set_nozzle_vacuum, set_nozzle_air)
- [ ] Test with mock boards

### Phase 3: API Endpoints (Week 2)
- [ ] Add `/api/nozzle/{name}/move-to-camera` endpoint
- [ ] Add `/api/nozzle/{name}/move-camera-here` endpoint
- [ ] Add `/api/nozzle/{name}/vacuum` endpoint
- [ ] Add `/api/nozzle/{name}/air` endpoint
- [ ] Update `/api/status` to include nozzle state
- [ ] All endpoints routed through job system with "nozzle" domain

### Phase 4: Frontend (Week 2)
- [ ] Add nozzle card UI (per nozzle: position, valve toggles, sync buttons)
- [ ] Implement `moveNozzleToCamera()`, `moveCameraToNozzle()` JavaScript helpers
- [ ] Add valve toggle buttons with state feedback
- [ ] Job tracking for nozzle operations

### Phase 5: Testing (Week 3)
- [ ] Offset math validation tests
- [ ] Mock hardware tests for valve commands
- [ ] End-to-end integration tests (API → driver → board)
- [ ] UI interaction tests

---

## 6. Backward Compatibility

### Risk: Breaking changes?
- ✅ **Config schema**: Existing `nozzles` config stays valid (add optional `offset_x`, `offset_y`, valve config)
- ✅ **PositionStore**: No changes to public API
- ✅ **HardwareDriver**: Adding methods, not replacing (backward compatible)
- ✅ **API endpoints**: New endpoints don't conflict with existing ones
- ✅ **SerialBoard**: IO abstraction is purely additive

### Migration Path
1. Deploy updated config schema (backward compatible)
2. Load default offsets (0, 0) if not specified (no-op for existing systems)
3. Valve endpoints available but return "valve not configured" if missing
4. Existing coord/head/light endpoints unchanged

---

## 7. Conclusion & Next Steps

### Summary of Improvements Needed
1. **Config schema** — add nozzle offsets, valve hardware mapping
2. **ValveStore** — new state container for valve control
3. **NozzleConfig** — type-safe nozzle geometry model
4. **HardwareDriver** — nozzle motion + valve control methods
5. **API endpoints** — 4 new endpoints + status extension
6. **IO abstraction** — extend SerialBoard with generic pin control
7. **Frontend** — nozzle card + sync buttons + valve toggles
8. **Domain conflict policy** — add "nozzle" domain to auto-cancel logic
9. **Module organization** — prepare for future CameraVisionModule split (defer actual split)

### Recommended Start
1. **THIS WEEK**: Implement ValveStore, NozzleConfig, update system.json schema
2. **NEXT WEEK**: Add HardwareDriver methods, API endpoints, frontend UI
3. **WEEK AFTER**: Integration testing, real hardware validation

### Post-Feature Improvements (DEFER)
- CameraVisionModule refactor (split into micro-controllers)
- Cross-board motion error recovery
- Serial disconnect detection + board health events
- Startup/shutdown orchestration guards

---

**Architecture is sound for placement head feature. Proceed with Phase 1 implementation.**
