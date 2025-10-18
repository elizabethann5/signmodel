import { useEffect, useRef } from 'react';
import { Volume2, VolumeX } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';

interface TranslationDisplayProps {
  text: string;
  isPlaying: boolean;
  onPlayAudio: () => void;
  onStopAudio: () => void;
}

export const TranslationDisplay = ({ 
  text, 
  isPlaying, 
  onPlayAudio, 
  onStopAudio 
}: TranslationDisplayProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom when text updates
    if (scrollRef.current) {
      const scrollElement = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [text]);

  return (
    <div className="flex flex-col h-full bg-card rounded-lg shadow-[var(--shadow-card)] border border-border">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <h2 className="text-lg font-semibold text-foreground">Translation Output</h2>
        {text && (
          <Button
            size="sm"
            variant="ghost"
            onClick={isPlaying ? onStopAudio : onPlayAudio}
            className="hover:bg-secondary"
          >
            {isPlaying ? (
              <>
                <VolumeX className="w-4 h-4 mr-2" />
                Stop Audio
              </>
            ) : (
              <>
                <Volume2 className="w-4 h-4 mr-2" />
                Play Audio
              </>
            )}
          </Button>
        )}
      </div>

      <ScrollArea ref={scrollRef} className="flex-1 p-6">
        {text ? (
          <div className="space-y-4">
            <p className="text-2xl leading-relaxed text-foreground font-medium">
              {text}
            </p>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-center">
            <div className="space-y-2">
              <p className="text-xl text-muted-foreground">No translation yet</p>
              <p className="text-sm text-muted-foreground/70">
                Start streaming to see real-time translations
              </p>
            </div>
          </div>
        )}
      </ScrollArea>
    </div>
  );
};
