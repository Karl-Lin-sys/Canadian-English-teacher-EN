import threading
import time
import asyncio
import os
import sys

# We use pygame for BOTH audio and the GUI window,
# since OpenCV's cv2.imshow can fail on some Windows setups if opencv-python-headless was installed.
try:
    import edge_tts
    import pygame
    CAN_SPEAK = True
except ImportError:
    CAN_SPEAK = False

class CanadianEnglishTeacherApp:
    def __init__(self):
        if not CAN_SPEAK:
            print("\n[WARNING] Please run: pip install edge-tts pygame")
            sys.exit(1)
            
        print("Initializing the Canadian English Teacher AI GUI (Pygame version)...")
        pygame.init()
        pygame.mixer.init()
        
        # Load the teacher's portrait image
        self.image_path = "teacher.jpg"
        try:
            self.base_image = pygame.image.load(self.image_path)
        except Exception:
            # Fallback if image is missing
            self.base_image = pygame.Surface((600, 600))
            self.base_image.fill((0, 0, 0))
            font = pygame.font.SysFont(None, 48)
            text = font.render("Avatar Image Missing", True, (255, 255, 255))
            self.base_image.blit(text, (100, 250))
            
        # Resize image if it's too large for standard screens
        rect = self.base_image.get_rect()
        if rect.width > 600:
            scale = 600 / rect.width
            self.base_image = pygame.transform.smoothscale(self.base_image, (600, int(rect.height * scale)))
            rect = self.base_image.get_rect()
            
        # Setup Pygame Window
        self.screen = pygame.display.set_mode((rect.width, rect.height))
        pygame.display.set_caption("Canadian English Teacher Avatar")
        
        # Set coordinates for the mouth based on the generated portrait
        self.mouth_center = (int(rect.width * 0.49), int(rect.height * 0.64))
        self.mouth_width = int(rect.width * 0.15)
        
        self.is_speaking = False
        self.mouth_open = False
        self.running = True

    def generate_canadian_response(self, user_input: str) -> str:
        """
        Simulates the LLM generating a response in Canadian English.
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
        voice = "en-CA-ClaraNeural" # Female voice
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

    def terminal_thread(self):
        """
        Runs in a background thread to handle terminal inputs without freezing the GUI.
        """
        print("\n" + "="*50)
        print("🍁 Welcome to the Canadian English Teacher Avatar! 🍁")
        print("Type a sentence you want to practice, or 'quit' to exit.")
        print("="*50)
        print("-> Please look at the Pygame window to see your teacher!")
        
        while self.running:
            try:
                # Blocks waiting for input
                user_input = input("\n📝 You: ")
                
                if not self.running:
                    break
                    
                if user_input.lower() in ['quit', 'exit']:
                    print("See you later, eh!")
                    self.running = False
                    break
                    
                if not user_input.strip():
                    continue
                
                # Generate AI response
                teacher_reply = self.generate_canadian_response(user_input)
                print(f"\n🍁 Teacher: {teacher_reply}")
                
                # Generate Audio via Edge TTS
                audio_file = "temp_response.mp3"
                asyncio.run(self._async_text_to_speech(teacher_reply, audio_file))
                
                # Trigger audio playback in Pygame
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # Tell the GUI to start animating
                self.is_speaking = True
                
                # Wait until audio finishes playing
                while pygame.mixer.music.get_busy() and self.running:
                    time.sleep(0.1)
                    
                # Stop animation
                self.is_speaking = False
                
                # Cleanup the audio file
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                    except:
                        pass

            except EOFError:
                self.running = False
                break
            except Exception as e:
                print(f"Error in terminal thread: {e}")
                self.running = False
                break

    def run(self):
        """
        The main GUI loop must run on the main thread for OS stability.
        """
        # Start the terminal input handling in the background
        t_thread = threading.Thread(target=self.terminal_thread, daemon=True)
        t_thread.start()
        
        clock = pygame.time.Clock()
        last_toggle_time = time.time()
        
        # Main GUI rendering loop
        while self.running:
            # Handle OS window events (prevents "Not responding" crash)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            # Draw the base portrait
            self.screen.blit(self.base_image, (0, 0))
            
            # Draw the dynamic mouth if speaking
            if self.is_speaking:
                current_time = time.time()
                # Toggle mouth open/closed every 150ms
                if current_time - last_toggle_time > 0.15:
                    self.mouth_open = not self.mouth_open
                    last_toggle_time = current_time
                    
                if self.mouth_open:
                    # Open mouth (Ellipse)
                    rect = pygame.Rect(0, 0, self.mouth_width, int(self.mouth_width * 0.3))
                    rect.center = self.mouth_center
                    pygame.draw.ellipse(self.screen, (40, 20, 30), rect)
                else:
                    # Closed mouth (Line)
                    start_pos = (self.mouth_center[0] - self.mouth_width // 2, self.mouth_center[1])
                    end_pos = (self.mouth_center[0] + self.mouth_width // 2, self.mouth_center[1])
                    pygame.draw.line(self.screen, (40, 20, 30), start_pos, end_pos, 4)
            
            # Update window
            pygame.display.flip()
            # Cap the framerate to 30 FPS to save CPU
            clock.tick(30)
            
        pygame.quit()

if __name__ == "__main__":
    app = CanadianEnglishTeacherApp()
    app.run()
