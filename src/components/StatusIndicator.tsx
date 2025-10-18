import { Wifi, WifiOff, Activity } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface StatusIndicatorProps {
  isConnected: boolean;
  isStreaming: boolean;
  fps?: number;
  latency?: number;
}

export const StatusIndicator = ({ 
  isConnected, 
  isStreaming, 
  fps = 0,
  latency = 0
}: StatusIndicatorProps) => {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <Badge 
        variant={isConnected ? "default" : "secondary"}
        className={`${
          isConnected 
            ? 'bg-primary/20 text-primary border-primary/30' 
            : 'bg-destructive/20 text-destructive border-destructive/30'
        } transition-all`}
      >
        {isConnected ? (
          <>
            <Wifi className="w-3 h-3 mr-1.5" />
            Connected
          </>
        ) : (
          <>
            <WifiOff className="w-3 h-3 mr-1.5" />
            Disconnected
          </>
        )}
      </Badge>

      {isStreaming && (
        <Badge 
          variant="default"
          className="bg-accent/20 text-accent border-accent/30"
        >
          <Activity className="w-3 h-3 mr-1.5" />
          Streaming
        </Badge>
      )}

      {isConnected && (
        <>
          <Badge variant="outline" className="border-border/50">
            {fps} FPS
          </Badge>
          {latency > 0 && (
            <Badge variant="outline" className="border-border/50">
              {latency}ms latency
            </Badge>
          )}
        </>
      )}
    </div>
  );
};
