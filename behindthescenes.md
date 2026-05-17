# Behind the scenes

It is kind of redicolous to write another S/W for an SMD pick and place machine.
After all, there is openPnP which seems to do a great job, at least for most of us.
However, I am - with my own build of a pnp-machine - struggling beyond my patience to get openPnP to work. This is mostly because I simply don't understand the behaviour of the coordinate system. If there was a book about openPnP which I could read through, I would do so and, most likely, understand what I don't undertstand clicking through page after page after page.
And - as the question is eminent - I do not want to dive into the S/W, simply because it's written in Java, a programming language I still haven't understood why it is still alive.

I do believe, however, that AI is here to stay. And that's why I have decided to start writing - or better better - orchestratimg my own interpretation of a pick and place machine S/W.

## Status

openSMT ist far away from being a working S/W. Up until now I got the backend working and some kind of frontend.
Backend???? Wait ... why backend???
Well, this is mostly due to my development setup. My machine is located in the basement, which is still some kind of construction zone (construction as in building construction) and doesn't have a convenient office desk, office chair ... you name it.
Thus, my machine is sitting in the basement with a wireless Reolink camera, and the controlling PC (Debain 13) and the Python written server.
My development machine - I love my M2 MacBook - is located where I sit most comfortable and can do most of the stuff remotely.

The frontend is also written in Python and uses Qt for the GUI part.

## Everything else

You figured .... this bloke us crazy and, what shall I say, you name it.
If you want to look into the crazy stuff, be my guest. Contact me if you like, share your thoughts.

## Milestones and Tollgates

I have no sharp milestones or tollgates. This is my personal venture and hobby.
I'm planning to have my first boards produced by end of September, provided I can get the layout ready ...

# Outlooks

As I need to switch from S/W to H/W and back once in a while, I'm working on automatic feeders ccasionally.