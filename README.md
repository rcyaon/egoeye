# egoeye

verifying human demonstrations with signal-integrity tricks. built in a one-day hackathon sprint on the [egoverse](https://github.com/GaTech-RL2/EgoVerse) robot-learning dataset, then hardened on the real data.

robots learn by watching videos of humans doing chores. the problem: some of those humans dropped things, fumbled, or had to retry — and the video looks fine, so those bad examples end up in the training data anyway. egoeye finds them without any AI.

the idea comes from chip design. engineers who build high-speed links have a whole toolbox for one question: *is this signal behaving consistently, and when exactly did it glitch?* they answer it with math. a person's wrist moving through a chore is also a signal — so egoeye points that same toolbox at humans:

- **a drop is a jolt.** smooth, intentional motion doesn't have sudden spikes. when something slips out of your hand, the trace spikes hard — the same trick used to hear a failing bearing inside a spinning machine.
- **a retry is fidgeting.** a clean task is a few big, confident motions; redoing a grasp is lots of small back-and-forth ones. counting cycles by size is borrowed from fatigue engineering, where it predicts when metal will crack.
- **consistency is an open eye.** this is the signature signal-integrity move. to check a cable carrying billions of bits, engineers overlay every bit on top of every other bit; consistent bits pile up into a clear opening — an "eye" — and a flaky link smears it shut. do the same with a repetitive chore: stack every scrub/fold/rinse cycle on top of each other. a clean demonstrator's cycles line up into a tight eye; hesitation and fumbling smear it. the fumbled cycle is literally the trace that pokes into the eye.
