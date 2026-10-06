Drop folder for the "Handpicked samples" section of the DEP1 web client.
- Put 4-5 stereo .wav files here (any name; the stem is shown as the title).
- Optional notes.json: {"<stem>": "<one-line note>"} -> shown next to each file.
- Served by the server as GET /api/samples and /samples/<name>.wav (INTERFACE.md section 5). No rebuild needed.
