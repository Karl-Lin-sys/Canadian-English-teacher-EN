import torch
import cv2
import numpy as np
import time
import asyncio
import os

# Try to import TTS libraries for actual voice generation
try:
    import edge_tts
    import pygame
    CAN_SPEAK = True
except ImportError:
    CAN_SPEAK = False

from avatar_model import AudioToFacialLandmarks

class CanadianEnglishTeacherApp:
    def __init__(self):
        print("Initializing the Canadian English Teacher AI...")
        # 1. Load the Avatar Model (The "Puppet")
        self.avatar_model = AudioToFacialLandmarks(audio_dim=256, expression_dim=68)
        self.avatar_model.eval() # Set to inference mode
        
        if not CAN_SPEAK:
            print("\n[WARNING] 'edge-tts' or 'pygame' is missing. Please run:")
            print("pip install edge-tts pygame")
            print("to enable real voice output.\n")
        
    def generate_canadian_response(self, user_input: str) -> str:
        """
        Simulates the LLM (The "Smart Brain") generating a response in Canadian English.
        """
        lower_input = user_input.lower()
        if "hello" in lower_input or "hi" in lower_input:
            return "Hello there! In Canada, we might say 'How's it goin, eh?' Give it a try!"
        elif "thank you" in lower_input or "thanks" in lower_input:
            return "You're very welcome! No worries at all."
        elif "sorry" in lower_input:
            return "Oh, sorry about that! Did you know Canadians are famous for apologizing?"
        elif "weather" in lower_input:
            return "Make sure to grab your toque, it's getting chilly outside!"
        elif "poutine" in lower_input:
            return "Oh, poutine is delicious! It's fries with cheese curds and gravy."
        else:
            return f"That's interesting! If you want to sound Canadian, just add 'eh' at the end, like this: {user_input}, eh?"

    async def _async_text_to_speech(self, text: str, output_file: str):
        """
        Uses Microsoft Edge's Neural TTS to generate high-quality Canadian English speech.
        """
        # en-CA-LiamNeural is a male Canadian voice. en-CA-ClaraNeural is female.
        voice = "en-CA-LiamNeural" 
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

    def text_to_speech(self, text: str) -> torch.Tensor:
        """
        Plays the TTS audio out loud using pygame and returns dummy features for the avatar.
        """
        print(f"🗣️  [Voice Box] Speaking: '{text}'")
        
        if CAN_SPEAK:
            audio_file = "temp_response.mp3"
            
            # Generate the audio file
            asyncio.run(self._async_text_to_speech(text, audio_file))
            
            # Play the audio file out loud!
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait until the voice finishes speaking
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            pygame.mixer.quit()
            
            # Clean up the temporary audio file
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except PermissionError:
                    pass # Windows might still hold a lock briefly
        else:
            # Fallback if libraries aren't installed
            time.sleep(1.5)
        
        # Return dummy audio features representing the speech
        return torch.randn(1, 256, 50)

    def generate_avatar_video(self, audio_features: torch.Tensor):
        """
        Uses the PyTorch model to generate lip-sync expressions based on the audio.
        """
        print("🎬 [Avatar] Calculating 3D facial expressions from audio...")
        with torch.no_grad():
            expressions = self.avatar_model(audio_features)
        
        print(f"✅ [Avatar] Generated {expressions.shape[1]} frames of animation.")

    def run(self):
        print("\n" + "="*50)
        print("🍁 Welcome to the Canadian English Teacher Avatar! 🍁")
        print("Type a sentence you want to practice, or 'quit' to exit.")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n📝 You: ")
                if user_input.lower() in ['quit', 'exit']:
                    print("See you later, eh!")
                    break
                if not user_input.strip():
                    continue
                
                # Step 1: Brain (LLM)
                teacher_reply = self.generate_canadian_response(user_input)
                print(f"\n🍁 Teacher: {teacher_reply}")
                
                # Step 2: Voice Box (TTS) plays out loud
                audio_features = self.text_to_speech(teacher_reply)
                
                # Step 3: Avatar (Lip Sync)
                self.generate_avatar_video(audio_features)
                
            except KeyboardInterrupt:
                print("\nSee you later, eh!")
                break

if __name__ == "__main__":
    app = CanadianEnglishTeacherApp()
    app.run()
