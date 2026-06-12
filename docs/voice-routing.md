# Voice Routing

HFabric uses its native in-process RVC engine for realtime voice conversion.
The Voice tab selects sounddevice input/output devices, applies native engine
settings, and starts or stops the live session through `/api/voice/engine/*`.

## Windows Output Setup

To use the converted voice in Discord, OBS, games, or meeting apps:

1. Install a virtual audio cable such as VB-CABLE, or use VoiceMeeter if you need
   more routing.
2. In the HFabric Voice tab, set `Input` to your physical microphone.
3. Set `Output` to the virtual cable input device.
4. In the target app, set the microphone to the matching virtual cable output.
5. Use `Monitor` only when you want to hear the converted voice locally.
6. Turn on live mode in the Voice tab.

Keep monitoring off when routing into apps unless you specifically need it;
otherwise you may hear doubled audio.

## Notes

- Required pretrain assets live under `models/voice/pretrain`.
- RVC voice slots live under `models/voice`.
- `Sample rate`, `Chunk`, crossfade, buffer, device, and gain settings are sent
  to the native voice engine.
- A live voice session owns the GPU while it is active. The arbiter frees any
  resident model before the session starts, and queued image/LLM jobs stay parked
  until the session stops.
