import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

// Mock AI service response - replace with actual external AI service integration
const mockAITranslation = (frameData: string): { translation: string; confidence: number; gesture_id: number } => {
  const samplePhrases = [
    "Hello, how are you?",
    "Thank you very much",
    "Nice to meet you",
    "Good morning",
    "Have a great day",
    "See you later",
    "I understand",
    "Please help me",
  ];
  
  const randomPhrase = samplePhrases[Math.floor(Math.random() * samplePhrases.length)];
  
  return {
    translation: randomPhrase,
    confidence: 0.85 + Math.random() * 0.15, // 0.85 to 1.0
    gesture_id: Math.floor(Math.random() * 100),
  };
};

// Process frame and get translation from AI service
const processFrame = async (frameData: string): Promise<{ translation: string; confidence: number; gesture_id: number }> => {
  // TODO: Replace with actual external AI service call
  // const response = await fetch('https://ai.external.service/api/v1/translate', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({ frame: frameData }),
  // });
  // const result = await response.json();
  // return result;

  // For now, return mock data
  await new Promise(resolve => setTimeout(resolve, 20 + Math.random() * 30)); // Simulate 20-50ms latency
  return mockAITranslation(frameData);
};

serve(async (req) => {
  console.log('WebSocket connection request received');
  
  const upgrade = req.headers.get("upgrade") || "";
  if (upgrade.toLowerCase() !== "websocket") {
    return new Response("Expected WebSocket connection", { status: 426 });
  }

  try {
    const { socket, response } = Deno.upgradeWebSocket(req);

    socket.onopen = () => {
      console.log("WebSocket connection established");
      socket.send(JSON.stringify({
        type: "connected",
        message: "Connected to Auralis translation service",
      }));
    };

    socket.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received message type:', data.type);

        if (data.type === "frame" && data.frame) {
          const requestTimestamp = data.timestamp || Date.now();
          
          // Process the frame through AI service
          const result = await processFrame(data.frame);
          
          const responseTimestamp = Date.now();
          const latency = responseTimestamp - requestTimestamp;
          
          // Send translation back to client
          socket.send(JSON.stringify({
            type: "translation",
            translation: result.translation,
            confidence: result.confidence,
            gesture_id: result.gesture_id,
            latency: latency,
            timestamp: responseTimestamp,
          }));

          console.log(`Translation sent: "${result.translation}" (latency: ${latency}ms)`);
        }
      } catch (error) {
        console.error("Error processing message:", error);
        socket.send(JSON.stringify({
          type: "error",
          data: error instanceof Error ? error.message : "Unknown error occurred",
        }));
      }
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    socket.onclose = () => {
      console.log("WebSocket connection closed");
    };

    return response;
  } catch (error) {
    console.error("WebSocket upgrade error:", error);
    return new Response("Failed to upgrade connection", { status: 500 });
  }
});
