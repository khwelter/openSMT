# openSMT
Open-source software for SMD pick and place machines.
## License
MIT License - See LICENSE file for details.
## Requirements
- Python 3.11+
- Qt 6
- OpenCV
- See `requirements.txt` for full dependencies
## Why openSMT
This is a very good question, especially since there is an apparently working system called openPnP.
The honest answer is ... openPnP is about to drive me nuts.
I have a working mechanics, with a rather high precision. I believe I can verify this with various tools under Linux. With a working area of about 500x500 mm I'm hitting the 500x500 mm position - or any other arbitrary position - with pretty much no deviation after any homing cycle at any given speed.
When I run the whole mechanics under openPnP, the homing fiducial is always off, sometimes by 2mm, sometimes by -1mm, but - again - always off. Readjusting the homing fiducials position doesn't change anything. It's simply not logical. Running an Opulo Lumen in the company I work for works flawlessly, though we are facing similar issues.

Given all of that, plus the availability of AI, I've decided to give another open* project another shot, this time it's gonna be called openSMT.

openSMT is gonna be written in Python. I personally am not a declared Python lover, but I believe to have learned that AI loves Python. So will I - meanwhile this venture.

openSMT will need a lot of good image processing, which is gonna be based on openCV, a platform which I used already about 10 years ago while building a simple image recognition system for plastics industry.

Here we go. If you wanny join ... be my guest, or fellow worker.

## Installation
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
