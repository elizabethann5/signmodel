import { useState, useEffect, useCallback, useRef } from 'react';
import { useToast } from '@/hooks/use-toast';
import { io, Socket } from 'socket.io-client';

interface TranslationResponse {
  text: string;
  audio?: string;
  timestamp?: number;
}

interface StatusResponse {
  message: string;
}

interface ErrorResponse {
  message: string;
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
  const socketRef = useRef<Socket | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    // Connect to Socket.IO server
    const connect = () => {
      try {
        const socket = io(url, {
          transports: ['websocket', 'polling']
        });
        socketRef.current = socket;

        socket.on('connect', () => {
          console.log('Socket.IO connected');
          setIsConnected(true);
          toast({
            title: "Connected",
            description: "Connected to sign language translation server",
          });
        });

        socket.on('status', (data: StatusResponse) => {
          console.log('Status received:', data);
          toast({
            title: "Server Status",
            description: data.message,
          });
        });

        socket.on('translation_output', (data: TranslationResponse) => {
          console.log('Translation received:', data);
          if (data.text) {
            setLastTranslation(data.text);
          }
          // Calculate latency if timestamp is provided
          if (data.timestamp) {
            const currentTime = Date.now();
            const calculatedLatency = currentTime - data.timestamp;
            setLatency(calculatedLatency);
          }
        });

        socket.on('error', (data: ErrorResponse) => {
          console.error('Server error:', data);
          toast({
            title: "Translation Error",
            description: data.message || "An error occurred",
            variant: "destructive",
          });
        });

        socket.on('disconnect', () => {
          console.log('Socket.IO disconnected');
          setIsConnected(false);
          setLatency(0);
          toast({
            title: "Disconnected",
            description: "Lost connection to translation server",
            variant: "destructive",
          });
        });

        socket.on('connect_error', (error) => {
          console.error('Socket.IO connection error:', error);
          toast({
            title: "Connection Failed",
            description: "Could not connect to translation server",
            variant: "destructive",
          });
        });
      } catch (error) {
        console.error('Failed to create Socket.IO connection:', error);
        toast({
          title: "Connection Failed",
          description: "Could not establish connection to server",
          variant: "destructive",
        });
      }
    };

    connect();

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [url, toast]);

  const sendFrame = useCallback((frameData: string) => {
    if (socketRef.current && socketRef.current.connected) {
      const timestamp = Date.now();
      socketRef.current.emit('video_frame_stream', {
        frame: frameData,
        timestamp,
      });
    }
  }, []);

  return {
    isConnected,
    sendFrame,
    latency,
    lastTranslation,
  };
};
