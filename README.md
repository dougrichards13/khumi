# Project Khumi

A voice-first, offline educational companion for single-board computers.

Khumi (pronounced "koo-me") is a learning friend that runs entirely on a Raspberry Pi with no internet required. It combines offline encyclopedic content (Wikipedia, Gutenberg, LibreTexts, and more) with a local LLM to create a patient, always-available teacher for anyone, anywhere.

## Why

Imagine a child tending a flock in a remote village, learning about photosynthesis from a handheld device that speaks to them. No internet, no subscription, no data collection. Just a $100 box with 500GB of human knowledge and an AI that knows how to teach from it.

Khumi is not a chatbot. It is not entertainment. It is a **teaching engine** — given context from its library, it explains, teaches, quizzes, and encourages.

## Architecture

```
[Voice/Text] → [Kiwix Search] → [ZIM Articles]
  → [llama.cpp LLM] → [Cited Answer] → [Piper TTS Voice]
```

- **Content**: Internet-in-a-Box (IIAB) + Kiwix serving ZIM files (Wikipedia, Gutenberg, LibreTexts, StackOverflow, etc.)
- **LLM**: llama.cpp server running quantized GGUF models natively on ARM (no GPU required)
- **Voice**: Piper TTS (text-to-speech) + faster-whisper (speech-to-text)
- **UI**: Mobile-first web app with onboarding, 4 learning modes, adaptive themes
- **Zero cloud dependency** after initial setup

## Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Board | Raspberry Pi 5 (8GB) | Raspberry Pi 5 (16GB) |
| Storage | 256GB NVMe | 1TB NVMe |
| Accelerator | None (CPU-only) | Hailo AI HAT+ 2 ($130) |
| Display | Any browser on LAN | 7" touchscreen |

### Performance (Pi 5, CPU-only, qwen2.5 0.5B Q4_K_M)

- Ask mode: ~5 seconds end-to-end
- Teach mode: ~60 seconds (generates structured lesson)
- TTS: 0.38x real-time (faster than speech)
- STT: Real-time on "tiny" model

With Hailo AI HAT+ 2: ~9.5 tok/s, sub-2s first token.

## Quick Start

```bash
# On a Pi 5 with IIAB already installed:
git clone https://github.com/dougrichards13/khumi.git
cd khumi
sudo bash scripts/install-khumi.sh
```

## Learning Modes

- **Ask Me** — Ask any question, get a concise cited answer
- **Teach Me** — Structured mini-lessons (Hook → Key Concepts → Example → Summary)
- **Quiz Me** — Multiple-choice questions generated from content
- **Settings** — Theme (Daylight/Evening/Night/Power Save), text size, voice controls

## Content Library (56 ZIM files, 357 GB)

- Wikipedia English (116 GB)
- Project Gutenberg — 70,000+ books (207 GB)
- LibreTexts — Biology, Chemistry, Physics, Math, Engineering, Medicine, and more
- Simple English Wikipedia
- iFixit repair guides
- StackOverflow, ServerFault, SuperUser
- WikiNews, WikiVoyage, WikiQuote, WikiVersity
- DevDocs (Python, JavaScript, Rust, Go, C++, and more)

## Project Status

**Phase 1** — Core pipeline working (search → LLM → answer → voice)
**Phase 2** — Kiosk mode, install automation, multi-device testing
**Phase 3** — Distilled Khumi model, federated learning, particle UI

## License

MIT

## Contributing

This project aims to be free to the world. Contributions welcome.

Co-Authored-By: Oz <oz-agent@warp.dev>
