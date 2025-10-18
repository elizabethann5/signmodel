import { useEffect, useRef, useState } from 'react';
import { Camera, CameraOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface VideoFeedProps {
  onFrameCapture: (frameData: string) => void;
  isStreaming: boolean;
  captureRate?: number; // FPS
}

export const VideoFeed = ({ onFrameCapture, isStreaming, captureRate = 10 }: VideoFeedProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hasCamera, setHasCamera] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const { toast } = useToast();

  const startCamera = async () => {
    setIsLoading(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setHasCamera(true);
        toast({
          title: "Camera Enabled",
          description: "Video feed is now active",
        });
      }
    } catch (error) {
      console.error('Camera access error:', error);
      toast({
        title: "Camera Access Denied",
        description: "Please grant camera permissions to use the translator",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setHasCamera(false);
      toast({
        title: "Camera Disabled",
        description: "Video feed stopped",
      });
    }
  };

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current || !hasCamera) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (context && video.readyState === video.HAVE_ENOUGH_DATA) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert to base64
      const frameData = canvas.toDataURL('image/jpeg', 0.8);
      const base64Data = frameData.split(',')[1];
      onFrameCapture(base64Data);
    }
  };

  useEffect(() => {
    if (isStreaming && hasCamera) {
      const intervalMs = 1000 / captureRate;
      captureIntervalRef.current = setInterval(captureFrame, intervalMs);
    } else {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    }

    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, [isStreaming, hasCamera, captureRate]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="relative w-full h-full flex flex-col">
      <div className="relative flex-1 bg-muted rounded-lg overflow-hidden shadow-[var(--shadow-card)]">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        <canvas ref={canvasRef} className="hidden" />
        
        {!hasCamera && (
          <div className="absolute inset-0 flex items-center justify-center bg-card/90 backdrop-blur-sm">
            <div className="text-center space-y-4">
              <CameraOff className="w-16 h-16 mx-auto text-muted-foreground" />
              <p className="text-lg text-muted-foreground">Camera not active</p>
            </div>
          </div>
        )}

        {isStreaming && (
          <div className="absolute top-4 right-4 px-3 py-1 bg-accent/90 backdrop-blur-sm rounded-full flex items-center gap-2">
            <div className="w-2 h-2 bg-background rounded-full animate-pulse" />
            <span className="text-xs font-medium text-accent-foreground">LIVE</span>
          </div>
        )}
      </div>

      <div className="mt-4 flex gap-2">
        {!hasCamera ? (
          <Button 
            onClick={startCamera} 
            disabled={isLoading}
            className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground"
          >
            <Camera className="w-4 h-4 mr-2" />
            {isLoading ? 'Starting...' : 'Enable Camera'}
          </Button>
        ) : (
          <Button 
            onClick={stopCamera} 
            variant="secondary"
            className="flex-1"
          >
            <CameraOff className="w-4 h-4 mr-2" />
            Disable Camera
          </Button>
        )}
      </div>
    </div>
  );
};
