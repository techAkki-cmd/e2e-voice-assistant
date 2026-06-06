# JarvisLabs Real-Time Voice Assistant

JarvisLabs Real-Time Voice Assistant is an end-to-end spoken AI assistant built with open models. A user speaks into the browser, the system streams audio through ASR, retrieves company knowledge when useful, reasons with Qwen, and streams synthesized speech back to the browser.

## Architecture

```mermaid
flowchart LR
    user["User Browser<br/>Mic + Speaker<br/>Frontend Observatory UI"]

    subgraph client["Client Layer"]
        user
        audioWorklet["AudioWorklet<br/>16 kHz PCM16 Frames"]
        player["Streaming Audio Playback<br/>PCM Response Chunks"]
    end

    subgraph backend["Spring Boot WebFlux Orchestrator"]
        ws["WebSocket Endpoint<br/>Binary/Text Multiplexing<br/>/api/v1/audio/stream"]
        router["Session Router<br/>Correlation ID + User ID<br/>Trace Propagation"]
        liveListener["ASR Live Transcript Listener"]
        audioListener["TTS Audio Response Listener"]
        bargeIn["Barge-in Control Handler"]
    end

    subgraph broker["RabbitMQ Message Broker"]
        qAudioIn["audio.incoming.raw<br/>Raw PCM Frames"]
        qAsrLive["text.asr.live<br/>Partial/Final Transcript Events"]
        qRag["text.rag.processing<br/>Final Transcript for Retrieval"]
        qLlm["text.llm.processing<br/>Transcript + Retrieved Context"]
        qTts["text.tts.processing<br/>LLM Text Chunks"]
        qAudioOut["audio.outgoing.stream<br/>Synthesized PCM Audio"]
        qControl["control.signals<br/>Fanout Exchange<br/>Barge-in / Kill Response"]
    end

    subgraph ai["Python GPU Inference Workers"]
        asr["ASR Worker<br/>WebRTCVAD Mode 1<br/>3.0x Digital Gain for VAD<br/>Streaming Sherpa + Whisper Final"]
        rag["RAG Worker<br/>SentenceTransformers Embeddings<br/>Low-confidence Miss Sanitization"]
        llm["LLM Worker<br/>Qwen2.5-3B-Instruct<br/>Jarvis Hybrid Prompt<br/>Streaming Tokens"]
        tts["TTS Worker<br/>MeloTTS<br/>Clause-level Streaming Synthesis"]
    end

    subgraph state["Stateful Services"]
        pg["pgvector / PostgreSQL<br/>Company Knowledge Embeddings"]
        redis["Redis<br/>voice:history:{user_id}<br/>Conversation Memory + TTL"]
    end

    user --> audioWorklet
    audioWorklet -->|"WebSocket binary PCM frames"| ws
    ws --> router
    router --> qAudioIn
    router -->|"text control message"| bargeIn
    bargeIn --> qControl

    qAudioIn --> asr
    asr --> qAsrLive
    asr --> qRag

    qAsrLive --> liveListener
    liveListener -->|"WebSocket text transcript event"| ws

    qRag --> rag
    rag <--> pg
    rag --> qLlm

    qLlm --> llm
    llm <--> redis
    llm --> qTts
    llm -. "interrupt-aware generation" .-> qControl

    qTts --> tts
    qControl --> tts
    qControl --> llm
    tts --> qAudioOut

    qAudioOut --> audioListener
    audioListener -->|"WebSocket binary PCM audio"| ws
    ws --> player
    player --> user
```

## Runtime Components

- Browser frontend streams microphone PCM frames over WebSocket and plays response PCM audio.
- Spring Boot WebFlux orchestrator multiplexes binary audio, transcript events, response audio, and barge-in control messages.
- RabbitMQ decouples audio ingress, ASR live transcripts, RAG requests, LLM chunks, TTS synthesis, response audio, and interruption signals.
- Python GPU workers run ASR, RAG, LLM, and TTS independently.
- pgvector stores embedded company knowledge for grounded support answers.
- Redis stores conversation history as `voice:history:{user_id}` with TTL-backed memory.
