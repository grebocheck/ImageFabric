# Voice Routing

HFabric wraps w-okada / MMVCServerSIO for realtime voice conversion. w-okada owns
the audio stream; HFabric selects its devices and starts/stops the stream.

## Windows output setup

To use the converted voice in Discord, OBS, games, or meeting apps:

1. Install a virtual audio cable such as VB-CABLE, or use VoiceMeeter if you need
   more routing.
2. In the HFabric Voice tab, start the w-okada server.
3. Set `Input` to your physical microphone.
4. Set `Output` to the virtual cable input device.
5. In the target app, set the microphone to the matching virtual cable output.
6. Use `Monitor` only when you want to hear the converted voice locally.

Keep monitoring off when routing into apps unless you specifically need it;
otherwise you may hear doubled audio.

## Notes

- `Sample rate`, `Chunk`, and gains are sent to w-okada through its
  `/update_settings` API.
- Live voice uses the GPU outside HFabric's worker. While it is active, queued
  image and LLM jobs wait instead of competing for VRAM.
- If audio does not start, open the w-okada UI once and confirm the same devices
  work there; HFabric is driving the same server settings.
