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

class AvatarGUI:
    def __init__(self):
        # Load the generated Canadian Teacher portrait
        self.image_path = "teacher.jpg"
        self.base_image = cv2.imread(self.image_path)
        
        if self.base_image is None:
            # Fallback black image if teacher.jpg is missing
            self.base_image = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(self.base_image, "Avatar Image Missing", (100, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            self.mouth_center = (256, 350)
            self.mouth_width = 60
        else:
            # Resize for consistent and visible display
            h, w = self.base_image.shape[:2]
            if w > 600:
                scale = 600 / w
                self.base_image = cv2.resize(self.base_image, (600, int(h * scale)))
            
            # Detect face to approximate mouth position using OpenCV Haar Cascades
            gray = cv2.cvtColor(self.base_image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                # Find the largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                # Approximate mouth position (bottom quarter of the face bounding box)
                self.mouth_center = (fx + fw//2, fy + int(fh * 0.8))
                self.mouth_width = int(fw * 0.35)
            else:
                # Fallback if detection fails
                h, w = self.base_image.shape[:2]
                self.mouth_center = (w//2, int(h*0.7))
                self.mouth_width = 40
                
        self.window_name = "Canadian English Teacher Avatar"
        cv2.imshow(self.window_name, self.base_image)
        cv2.waitKey(1)
        
    def show_idle(self):
        cv2.imshow(self.window_name, self.base_image)
        cv2.waitKey(1)

    def draw_speaking_frame(self, mouth_open: bool):
        """
        Dynamically draws an open/closed mouth on the static image to simulate talking.
        In a full Deep Learning app (like SadTalker), this is replaced by the 3DMM renderer.
        """
        img_copy = self.base_image.copy()
        if mouth_open:
            # Draw an open mouth
            cv2.ellipse(img_copy, self.mouth_center, (self.mouth_width//2, int(self.mouth_width*0.25)), 0, 0, 360, (20, 10, 20), -1)
        else:
            # Draw a closed mouth (subtle line)
            cv2.line(img_copy, (self.mouth_center[0] - self.mouth_width//2, self.mouth_center[1]), 
                               (self.mouth_center[0] + self.mouth_width//2, self.mouth_center[1]), (20, 10, 20), 3)
        
        cv2.imshow(self.window_name, img_copy)
        cv2.waitKey(1)


class CanadianEnglishTeacherApp:
    def __init__(self):
        print("Initializing the Canadian English Teacher AI GUI...")
        self.gui = AvatarGUI()
        if not CAN_SPEAK:
            print("\n[WARNING] Please run: pip install edge-tts pygame")
        
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
        # Using a Female Canadian voice to match the generated portrait
        voice = "en-CA-ClaraNeural" 
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

    def text_to_speech_and_animate(self, text: str):
        print(f"\n🍁 Teacher: {text}")
        
        if CAN_SPEAK:
            audio_file = "temp_response.mp3"
            asyncio.run(self._async_text_to_speech(text, audio_file))
            
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            mouth_open = False
            # Animation loop: updates the GUI in sync with the audio playback
            while pygame.mixer.music.get_busy():
                self.gui.draw_speaking_frame(mouth_open)
                cv2.waitKey(150) # Toggle speed (approx 6-7 fps for mouth movement)
                mouth_open = not mouth_open
                
            pygame.mixer.quit()
            self.gui.show_idle()
            
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except PermissionError:
                    pass
        else:
            # Fallback if audio libraries are missing
            for _ in range(10):
                self.gui.draw_speaking_frame(True)
                cv2.waitKey(150)
                self.gui.draw_speaking_frame(False)
                cv2.waitKey(150)
            self.gui.show_idle()

    def run(self):
        print("\n" + "="*50)
        print("🍁 Welcome to the Canadian English Teacher Avatar! 🍁")
        print("Type a sentence you want to practice, or 'quit' to exit.")
        print("="*50)
        print("-> Please look at the popup window to see your teacher!")
        
        while True:
            try:
                user_input = input("\n📝 You: ")
                if user_input.lower() in ['quit', 'exit']:
                    print("See you later, eh!")
                    break
                if not user_input.strip():
                    continue
                
                teacher_reply = self.generate_canadian_response(user_input)
                self.text_to_speech_and_animate(teacher_reply)
                
            except KeyboardInterrupt:
                print("\nSee you later, eh!")
                break
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = CanadianEnglishTeacherApp()
    app.run()
