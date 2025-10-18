import { useState, useEffect, useCallback, useRef } from 'react';
import { useToast } from '@/hooks/use-toast';

interface WebSocketMessage {
  type: 'translation' | 'error' | 'connected' | 'latency';
  data?: any;
  translation?: string;
  confidence?: number;
  gesture_id?: number;
  latency?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  sendFrame: (frameData: string) => void;
  latency: number;
  lastTranslation: string;
}

export const useWebSocket = (url: string): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [latency, setLatency] = useState(0);
  const [lastTranslation, setLastTranslation] = useState('');
  const wsRef = useRef<WebSocket | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    // Connect to WebSocket
    const connect = () => {
      try {
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('WebSocket connected');
          setIsConnected(true);
          toast({
            title: "Connected",
            description: "WebSocket connection established",
          });
        };

        ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            console.log('Received message:', message);

            switch (message.type) {
              case 'translation':
                if (message.translation) {
                  setLastTranslation(message.translation);
                }
                if (message.latency) {
                  setLatency(message.latency);
                }
                break;
              case 'latency':
                if (message.latency) {
                  setLatency(message.latency);
                }
                break;
              case 'error':
                console.error('WebSocket error:', message.data);
                toast({
                  title: "Translation Error",
                  description: message.data || "An error occurred",
                  variant: "destructive",
                });
                break;
            }
          } catch (error) {
            console.error('Error parsing message:', error);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          toast({
            title: "Connection Error",
            description: "Failed to maintain WebSocket connection",
            variant: "destructive",
          });
        };

        ws.onclose = () => {
          console.log('WebSocket disconnected');
          setIsConnected(false);
          setLatency(0);
        };
      } catch (error) {
        console.error('Failed to create WebSocket:', error);
        toast({
          title: "Connection Failed",
          description: "Could not establish WebSocket connection",
          variant: "destructive",
        });
      }
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url, toast]);

  const sendFrame = useCallback((frameData: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const timestamp = Date.now();
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        frame: frameData,
        timestamp,
      }));
    }
  }, []);

  return {
    isConnected,
    sendFrame,
    latency,
    lastTranslation,
  };
};
