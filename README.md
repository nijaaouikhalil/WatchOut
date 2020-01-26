# WatchOut



## Requirements
1. Python 3.6
2. OpenCV
3. PyTorch 0.4
4. TensorFlow==1.13.1
5. yolov3wights: https://pjreddie.com/media/files/yolov3.weights (put it the the Main folder)

Using PyTorch 0.3 will break the detector.



## Inspiration
We were thinking about how we could make our city roads safer. A lot of researches show that people tend to violate less the law when a cop is watching. In fact, Montreal used to use semi-trivial cops’ cars. Those cars showed great result when it comes to tickets. So, whenever people feel cops-free they tend to violate the law. We wanted everybody to feel the presence of the cops even when they can’t see their cars (which would lead to a safer roads!). This is why we decided to develop a product called WatchOut. 


## What it does

It essentially works like a traffic violation detector and reporter. In fact, we thought it would be a great if everyone can be a cop. We all agree that it’s annoying and dangerous to watch aggressive and violating traffic rules without being able to report it without losing a great amount of time and the driver might get a way with a simple consent. Instead of having to go through all that pain we decided to install a camera in the cars of some citizens. (or even taxi drivers since they move all around the city). This camera would detect traffic violations and automatically save the last 15s of the video before the violation into the cloud to finally send them to the SPVM, which then would be able to penalize them. Now it takes more than checking for cops to be ticket safe! Without knowing who has camera everyone would be more careful on roads which would decrease accidents caused by dangerous driving. Even more, those videos could be used by insurance companies and the city which would decrease court fees.
It is a great app that many would love to have.

## How we built it

Since we are a team of two members, we decided to separate the tasks into what everybody felt the more confident in. We decided to separate the job into 3 tasks.
Firstly, there is the first task where we had to train a model that detects traffic light stats (red, orange, green) and the back of cars. 
Secondly, whenever a car moves on red light, we save the last 15s of the video to a specific location (for this demo). This could be sent than to the government for further process.
Finally, logistics like the overall appearance of the app, the logo, the presentation and everything that revolves around that. The person in charge of this part was also leading the team in the sense that they were looking if everything was fitting together by making a todo list and other helpful tools like that to make sure we have everything we need.
We used mostly Python with typescript to create our app .
Challenges we ran into
We ran into a lot of challenges as we expected to. Even if we had some good bases on these programming languages, some of us didn't really work with that environment in the past, so we had to learn a lot of stuff to be able to build this app. We didn't have a lot of experience in those kinds of events.
Since we had to classify traffic lights, and cars, we needed to have access to a lot of pre-existing data to train our model. We found most of what we needed, but we couldn't get everything, so our algorithm can't take everything that we wanted in consideration.
Then we have the obvious, the debugging problems. Since we are not very experienced in that field, we had a lot of trouble doing it. Mostly because we didn't know very well the language we were working with, but also because debugging it is very complicated compared to other programming languages that we are used to.

## Accomplishments that I'm proud of

First, we think that our idea was great! This is the kind of app that a lot of people would like to use, and it has great potential in a market. This device might even be provided by insurance companies as it’s way of making our roads safer, which would lead to deceasing their compensation expenses. People would even ask for it as everyone would play his role to a safer city. So far, the app work only cars crossing the red light, but we can expand it to work on all traffic violations (it just needs more training). The scope of it touches everybody and it is great!
We are also proud of our team working. Everybody was here to help, and willing to take time off their project to help for another one. The patience of the people with more experience to show how to do a BUNCH of stuff was incredible.
We are very happy about the concept itself, it works well as a prototype. This is the kind of project that takes MONTHS to develop and we did a good part of it in a single night. We put a lot of efforts in the app and we are very proud of it. 

## What we learned

We learned how to separate the tasks well, and work with each other. We learned that even thought our expectations were high at the beginning of the competition we had to be more realistic with the implementation of the algorithm. We had to drop some part because we simply didn't have time for it. In that sense, we learned to focus on the more important things.

## What's next for WatchOut

The next step for WatchOut is to get a proper training to detect all traffic violations or even crimes!! We think that there is a good concept behind our project, and it has potential value in the insurance-gouverment market. And who knows? Maybe our sponsors might be interested!  :D 



