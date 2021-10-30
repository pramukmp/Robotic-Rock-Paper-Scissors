#Programme core
from cv2 import cv2
import hand_detection_module
from data_generate import num_hand
import pickle
from id_distance import calc_all_distance
import serial
import time

model_name = 'hand_model.sav'

play={"0":"Paper","1":"Rock","2":"Scissors"}

def timer(t):
  while t:
      mins, secs = divmod(t, 60)
      timer = '{:02d}:{:02d}'.format(mins, secs)
      print(timer, end="\r")
      time.sleep(1)
      t -= 1

def calculate_winner(move1, move2):
    if move1==play[move2]:
        return "Tie"

    if move1 == "Rock":
        if move2 == "2":
            return "User"
        if move2 == "0":
            return "Computer"

    if move1 == "Paper":
        if move2 == "1":
            return "User"
        if move2 == "2":
            return "Computer"

    if move1 == "Scissors":
        if move2 == "0":
            return "User"
        if move2 == "1":
            return "Computer"
    
             
# custom function
def rps(num):
  if num == 0:
    #ser.write(b'0')
    return 'Paper'
  elif num == 1:
    #ser.write(b'0')
    return 'Rock'
  elif num==2:
    #ser.write(b'0')
    return 'Scissors'


font = cv2.FONT_HERSHEY_PLAIN
hands = hand_detection_module.HandDetector(max_hands=num_hand)
model = pickle.load(open(model_name,'rb'))
cap = cv2.VideoCapture(0)
ser=serial.Serial('COM3',9600)

while cap.isOpened():
  # timer(5)
  # print("\n")
  
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue  

  TIMER=2
  prev = time.time()
  image=cv2.flip(frame,1)

  #time.sleep(2)
  while TIMER >= 0:
    if(TIMER==2):
      ser.write(b'1')
      #pass
    
    success,frame=cap.read()
    image = cv2.flip(frame, 1)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, str(TIMER),
                (280, 300), font,
                5.8, (0, 255, 0),
                4, cv2.LINE_AA)
    image = cv2.resize(image, (1600, 800))
    cv2.imshow('Hands', image)
    cv2.waitKey(125)

    # current time
    cur = time.time()

    # Update and keep track of Countdown
    # if time elapsed is one second
    # than decrease the counter
    if cur-prev >= 1:
      prev = cur
      TIMER = TIMER-1

  else:

    image, my_list = hands.find_hand_landmarks(cv2.flip(frame, 1),
                                              draw_landmarks=False)
    if my_list:      
      height, width, _ = image.shape
      all_distance = calc_all_distance(height,width, my_list)
      pred = rps(model.predict([all_distance])[0])

      ser.write(b'0')
      pos = (int(my_list[12][0]*height), int(my_list[12][1]*width))
      #image = cv2.putText(image,pred,pos,font,2,(0,0,255),2)
      rob=str(ser.readline()[1:-2],'utf-8')
      # rob="1"     
      robs="Robot: "+ play[rob]
      user="User : "+pred
      image=cv2.resize(image,(1600,800))
      image = cv2.rectangle(image, (450, 80), (1090, 200), (20,0,0), -1)

      image = cv2.putText(image, robs, (600, 600), font,1.8, (0, 0, 255), 4)
      image = cv2.putText(image, user, (600, 700), font, 1.8, (0, 0, 255), 4)
      winner = calculate_winner(pred,rob)
      winner = winner +" Wins !!" if winner != "Tie" else ".... TIE  ...."
      image = cv2.putText(image, winner, (600, 150), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2,cv2.LINE_AA)
      print(play[rob],'\n',calculate_winner(pred,rob),'wins')

    image = cv2.resize(image, (1600, 800))
    cv2.imshow('Hands', image)
  
  if cv2.waitKey(2000) == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    break

