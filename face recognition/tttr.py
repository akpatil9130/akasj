import tkinter as tk
from tkinter import Message, Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font



window=tk.Tk()

window.title('Face Recognation system')

window.geometry("1350x700+0+0")



lbl = tk.Label(window, text="Enter Roll no.", width=10, height=2, fg="yellow", bg="gray", font=('times', 15, 'bold'))
lbl.place(x=100, y=125)
txt1 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 25, 'bold'))
txt1.place(x=300, y=130)

lbl2 = tk.Label(window, text="Enter Name", width=10, height=2, fg="yellow", bg="gray", font=('times', 15, 'bold'))
lbl2.place(x=100, y=395)

txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 25, 'bold'))
txt2.place(x=300, y=400)


lblsub = tk.Label(window, text="Subject Name", width=10, height=2, fg="yellow", bg="gray", font=('times', 15, 'bold'))
lblsub.place(x=675, y=570)
txt3 = tk.Entry(window, width=15, bg="white", fg="black", font=('times', 25, 'bold'))
txt3.place(x=820, y=575)






def clear():
    txt1.delete(0,'end')
    res=""
    message.configure(text=res)

def clear2():
    txt2.delete(0,'end')
    res=""
    message.configure(text=res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError,ValueError):
     pass
    return False
def TakeImage():
    Id=(txt1.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadepath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadepath)
        sampleNum=0
        while(True):
            ret,img= cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                 cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                 sampleNum=sampleNum+1
                 cv2.imwrite("TrainingImage\ " +name +"."+Id +'.' + str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                 cv2.imshow('frame',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res ="Images saved for roll no:" +Id +" Name : "+ name
        row = [Id,name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if(is_number(Id)):
            res ="Enter Alphabetical Name"
            message.configure(text=res)

        if(name.isalpha()):
            res="Enter number Id"
            message.configure(text=res)


def TrainImage():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadepath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadepath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.write("TrainingImageLabel\Trainner.yml")
    res = "Student Details Add"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]
    for imagepath in imagepaths:
        pilImage=Image.open(imagepath).convert('L')
        imageNp =np.array(pilImage,'uint8')
        Id=int(os.path.split(imagepath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids



def TrackImage():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadepath = "haarcascade_frontalface_default.xml"
    facecascade = cv2.CascadeClassifier(harcascadepath)
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam=cv2.VideoCapture(0)
    font=cv2.FONT_HERSHEY_SIMPLEX
    col_names= ['Id','Name','Date','Time','  Subject',' Present']
    attendance = pd.DataFrame(columns = col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(gray, 1.2,5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if (conf <50):
                n = (txt3.get())
                pre = 'p'
                ts = time.time()
                date=datetime.datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id] ['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp,n,pre]


            else:
                Id='Unknown'
                tt=str(Id)
            if(conf>75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile)+".jpg",im[y:y+h,x:x+w])
            cv2.putText(im,str(tt),(x,y+h),font,1,(255,255,255),2)
        attendance= attendance.drop_duplicates(subset=['Id'],keep='first')
        cv2.imshow('im',im)
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_" +date+"_" + Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message.configure(text=res)



clearbutton = tk.Button(window, text="Clear", command=clear, fg="black", bg="gray", width=15, border=0, height=2,
                        activebackground="orange", font=('times', 15, 'bold'))
clearbutton.place(x=375, y=180)

clearbutton2 = tk.Button(window, text="Clear", command=clear2, fg="black", bg="gray", width=15, border=0, height=2,
                         activebackground="orange", font=('times', 15, 'bold'))
clearbutton2.place(x=375, y=450)




message = tk.Label(window, text="", bg="white", fg="red", width=50, height=12, activebackground="yellow",
                   font=('times', 15, 'bold'))
message.place(x=750, y=250)

lbl3 = tk.Label(window, text="Message", width=10, height=2, fg="yellow", bg="gray", font=('times', 20, 'bold'))
lbl3.place(x=990, y=175)




takeimg=tk.Button(window,text="Take Image",command=TakeImage,width=20,height=2,activebackground="orange",border='0',font=('Helvetica',15,'bold'))
takeimg.place(x=125,y=650)

trainimg=tk.Button(window,text="Add Details",command=TrainImage,width=20,height=2,activebackground="orange",border='0',font=('Helvetica',15,'bold'))
trainimg.place(x=475,y=650)

trackimg=tk.Button(window,text="Attendance",command=TrackImage,width=20,height=2,activebackground="orange",border='0',font=('Helvetica',15,'bold'))
trackimg.place(x=825,y=650)


takeimg=tk.Button(window,text="Quit",command=window.destroy,width=20,height=2,activebackground="orange",border='0',font=('Helvetica',15,'bold'))
takeimg.place(x=1200,y=650)





window.mainloop()