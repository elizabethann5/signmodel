import { Play, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ControlPanelProps {
  isStreaming: boolean;
  isConnected: boolean;
  onStartStreaming: () => void;
  onStopStreaming: () => void;
}

export const ControlPanel = ({
  isStreaming,
  isConnected,
  onStartStreaming,
  onStopStreaming,
}: ControlPanelProps) => {
  return (
    <div className="flex items-center justify-center gap-4">
      {!isStreaming ? (
        <Button
          size="lg"
          onClick={onStartStreaming}
          disabled={!isConnected}
          className="min-w-[200px] h-14 text-lg font-semibold bg-gradient-to-r from-primary to-accent hover:opacity-90 text-primary-foreground shadow-[var(--shadow-glow)]"
        >
          <Play className="w-5 h-5 mr-2" />
          Start Translation
        </Button>
      ) : (
        <Button
          size="lg"
          onClick={onStopStreaming}
          variant="secondary"
          className="min-w-[200px] h-14 text-lg font-semibold"
        >
          <Square className="w-5 h-5 mr-2" />
          Stop Translation
        </Button>
      )}
    </div>
  );
};
