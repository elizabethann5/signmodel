import { useState, useEffect } from 'react';
import { VideoFeed } from '@/components/VideoFeed';
import { TranslationDisplay } from '@/components/TranslationDisplay';
import { StatusIndicator } from '@/components/StatusIndicator';
import { ControlPanel } from '@/components/ControlPanel';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useTextToSpeech } from '@/hooks/useTextToSpeech';
import { Languages } from 'lucide-react';

const Index = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [translationText, setTranslationText] = useState('');
  const [fps] = useState(10); // Target FPS for frame capture

  // Socket.IO connection to Flask backend
  const wsUrl = 'http://localhost:5000';
  const { isConnected, sendFrame, latency, lastTranslation } = useWebSocket(wsUrl);
  const { speak, stop, isPlaying } = useTextToSpeech();

  // Update translation text when new translation arrives
  useEffect(() => {
    if (lastTranslation) {
      setTranslationText(prev => {
        const newText = prev ? `${prev} ${lastTranslation}` : lastTranslation;
        return newText;
      });
    }
  }, [lastTranslation]);

  const handleFrameCapture = (frameData: string) => {
    if (isStreaming && isConnected) {
      sendFrame(frameData);
    }
  };

  const handleStartStreaming = () => {
    setIsStreaming(true);
    setTranslationText('');
  };

  const handleStopStreaming = () => {
    setIsStreaming(false);
    stop();
  };

  const handlePlayAudio = () => {
    if (translationText) {
      speak(translationText);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-card">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Languages className="w-8 h-8 text-primary" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
                  Auralis
                </h1>
                <p className="text-sm text-muted-foreground">Real-Time Sign Language Translator</p>
              </div>
            </div>
            <StatusIndicator
              isConnected={isConnected}
              isStreaming={isStreaming}
              fps={fps}
              latency={latency}
            />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          {/* Video Feed */}
          <div className="h-[500px]">
            <VideoFeed
              onFrameCapture={handleFrameCapture}
              isStreaming={isStreaming}
              captureRate={fps}
            />
          </div>

          {/* Translation Display */}
          <div className="h-[500px]">
            <TranslationDisplay
              text={translationText}
              isPlaying={isPlaying}
              onPlayAudio={handlePlayAudio}
              onStopAudio={stop}
            />
          </div>
        </div>

        {/* Control Panel */}
        <div className="flex justify-center">
          <ControlPanel
            isStreaming={isStreaming}
            isConnected={isConnected}
            onStartStreaming={handleStartStreaming}
            onStopStreaming={handleStopStreaming}
          />
        </div>

        {/* Info Footer */}
        <div className="mt-8 p-6 bg-card/50 backdrop-blur-sm rounded-lg border border-border">
          <h3 className="text-lg font-semibold mb-2 text-foreground">How to Use</h3>
          <ol className="space-y-2 text-sm text-muted-foreground">
            <li>1. <strong className="text-foreground">Enable your camera</strong> - Grant camera permissions when prompted</li>
            <li>2. <strong className="text-foreground">Start translation</strong> - Click the "Start Translation" button</li>
            <li>3. <strong className="text-foreground">Sign</strong> - Perform sign language gestures in front of the camera</li>
            <li>4. <strong className="text-foreground">View translation</strong> - Real-time text appears in the translation panel</li>
            <li>5. <strong className="text-foreground">Play audio</strong> - Click the audio button to hear the translation</li>
          </ol>
        </div>
      </main>
    </div>
  );
};

export default Index;
