import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from geopy.geocoders import Nominatim
from twilio.rest import Client
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Twilio configuration
account_sid = ''  # Replace with your Twilio account SID in string format
auth_token =    '' # Replace with your Twilio auth token in string format
twilio_number =  '' # Replace with your Twilio phone number in string format
recipient_number = ''# Replace with the recipient's phone number in string format


# Email configuration
email_sender =  '' # Replace with your email address in string format
email_password = ''  # Replace with your email password or app-specific password in string format
email_recipient = ''  # Replace with the recipient's email address in string format

def load_images_from_folder(folder):
    images = []
    classNames = []
    myList = os.listdir(folder)
    for cl in myList:
        curImg = cv2.imread(f'{folder}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
    return images, classNames

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def get_location():
    # Use the specific latitude and longitude
    latitude = 19.1569
    longitude = 72.9936
    return latitude, longitude

def reverse_geocode(lat, lon):
    geolocator = Nominatim(user_agent="face_recognition_app")
    location = geolocator.reverse((lat, lon), language='en', exactly_one=True)
    if location:
        address = location.raw.get('address', {})
        address.update({'building': 'Jnan Vikas Mandal, Educational Trust, Degree College, Airoli'})
        building = address.get('building', '')
        road = address.get('road', '')
        street_number = address.get('house_number', '')
        sector_no = address.get('suburb', '') or address.get('neighbourhood', '') or address.get('city_district', '')
        city = address.get('city', '') or address.get('town', '') or address.get('village', '')
        state = address.get('state', '')
        country = address.get('country', '')
        return f"{building}, {street_number}, {road}, {sector_no}, {city}, {state}, {country}"
    else:
        return "Location not available"

def send_sms(name, address, datetime_detected):
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=f"Face detected: {name}\nAddress: {address}\nDate and Time: {datetime_detected}",
        from_=twilio_number,
        to=recipient_number
    )
    print(f"Message sent: {message.sid}")


def markRecord(name):
    latitude, longitude = get_location()
    address = reverse_geocode(latitude, longitude)
    now = datetime.now()
    datetime_detected = now.strftime('%Y-%m-%d %H:%M:%S')

    with open('ImageRec.csv', 'a') as f:
        f.writelines(f'\n{name},{datetime_detected},{address}')

    send_sms(name, address, datetime_detected)
    send_email(name, address, datetime_detected)


def send_email(name, address, datetime_detected):
    msg = MIMEMultipart()
    msg['From'] = email_sender
    msg['To'] = email_recipient
    msg['Subject'] = "Face Detection Alert"

    body = f"Face detected: {name}\nAddress: {address}\nDate and Time: {datetime_detected}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_sender, email_password)
        text = msg.as_string()
        server.sendmail(email_sender, email_recipient, text)
        server.quit()
        print(f"Email sent to {email_recipient}")
    except Exception as e:
        print(f"Failed to send email: {e}")


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition for Criminal Identification")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        self.label = Label(root, text="Criminal Identification", font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#333")
        self.label.pack(pady=20)

        self.btn_start = Button(root, text="Start Camera", command=self.start_camera, font=("Arial", 14), bg="#4CAF50", fg="#fff", activebackground="#45a049", activeforeground="#fff")
        self.btn_start.pack(pady=10)

        self.canvas = Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.btn_stop = Button(root, text="Stop Camera", command=self.stop_camera, font=("Arial", 14), bg="#f44336", fg="#fff", activebackground="#e53935", activeforeground="#fff")
        self.btn_stop.pack(pady=10)

        self.cap = None
        self.images, self.classNames = load_images_from_folder('ImageRec')
        self.encodeListKnown = findEncodings(self.images)
        print('Encoding Complete')

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Unable to open camera")
                self.cap = None
                return
        self.update_frame()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.canvas.delete("all")

    def update_frame(self):
        if self.cap is not None:
            success, frame = self.cap.read()
            if not success or frame is None:
                messagebox.showerror("Error", "Failed to capture image")
                self.stop_camera()
                return

            try:
                frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            except cv2.error as e:
                print("Error resizing frame:", e)
                return

            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            faces_cur_frame = face_recognition.face_locations(frame_rgb)
            encodes_cur_frame = face_recognition.face_encodings(frame_rgb, faces_cur_frame)

            for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
                matches = face_recognition.compare_faces(self.encodeListKnown, encode_face)
                face_dis = face_recognition.face_distance(self.encodeListKnown, encode_face)
                match_index = np.argmin(face_dis)

                if matches[match_index]:
                    name = self.classNames[match_index].upper()
                    y1, x2, y2, x1 = face_loc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markRecord(name)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=NW, image=imgtk)
            self.canvas.imgtk = imgtk

            self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
