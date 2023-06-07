# Compute the centre of the circle (ball) based on the sleap data (three points labelled by hand on random frames). 
import sleap as slp
import numpy as np

#----------------------------------------------------------------------------
# Function to compute the centre of the circle given three points on the edge. 
def CentreCircle(pt1, pt2, pt3): #pt are [x,y]
    '''
    Returns an array with the x and y coordinates of the centre of the circle

    General equation of the circle : 
    x^2 + y^2 + 2*a*x + 2*b*y + c = 0 

    as the points are on the circle we have the system : 
    x1^2 + y1^2 + 2*a*x1 + 2*b*y1 + c = 0 
    x2^2 + y2^2 + 2*a*x2 + 2*b*y2 + c = 0 
    x3^2 + y3^2 + 2*a*x3 + 2*b*y3 + c = 0 

    the unknows are a, b, c so we have : 
    [[2*x1, 2*y1, 1]      [[a]    [[-(x1^2 + y1^2)]
     [2*x2, 2*y2, 1]   *   [b] =   [-(x2^2 + y2^2)]
     [2*x3, 2*y3, 1]]      [c]]    [-(x3^2 + y3^2)]]

    finally we have :
    xc = -a
    yc = -b
    R = sqrt(xc^2 + yc^2 - c^2)  

    Note : 
    To have acess to the radius, please take of the # of the corresponding lines.
    Add the variable R in the return().
    '''
    M = np.array([[pt1[0], pt1[1], 1],
                  [pt2[0], pt2[1], 1],
                  [pt3[0], pt3[1], 1]])
    b = np.array([(pt1[0]**2 + pt1[1]**2), (pt2[0]**2 + pt2[1]**2), (pt3[0]**2 + pt3[1]**2)])

    sol = np.linalg.solve(M, b) # solve the system Mx = b 

    Xc, Yc = sol[0]/2, sol[1]/2
    R = np.sqrt((Xc - pt1[0])**2 + (Yc - pt1[1])**2)

    return(Xc, Yc, R)

#----------------------------------------------------------------------------
# Takes all the points from each labelled frame and put them in a numpy array. 
def LabelledPoints(labels):
    ''' 
    labels = the load of a sleap file

    Returns an array with all the points in arrays. 
    ''' 
    Points = []
    for i in range(len(labels)) :
        point = labels[i].instances[0].numpy()
        Points.append(point)

    return(Points)

#----------------------------------------------------------------------------
# Takes a set of data (sets of three points), compute the centre of the points and return a list of corresponding coordinates 
def Centres(labels):
    '''
    n_labelled_frames is the number of labelled frames.

    Returns an array of arrays with X,Y coordinates of the centre and the radius of the circles.
    '''
    Points = LabelledPoints(labels) 
    Centres = []
    for i in range(len(labels)):
        pt1 = Points[i][0]
        pt2 = Points[i][1]
        pt3 = Points[i][2]

        centre = CentreCircle(pt1, pt2, pt3)
        Centres.append(centre)

    return(Centres)

#----------------------------------------------------------------------------
# Save a file containing only the centres as data. /!\ rename the name of the file (date has to be updates) to be sure not to lose any data 
def SaveCentres(labels, name, date):
    '''
    Save a file containing the centres of the circle for each labelled frame
    
    name = [str] the name of the experiment. can be a number in a string form to sequences experiments 
    date = [int] formate AAMMJJ of the day the file was saved or the day the data were processed
    
    Returns nothing
    '''

    centre = Centres(labels)

    for i in range(len(labels)):
        labels[i].instances[0].points[0].x = centre[i][0]
        labels[i].instances[0].points[0].y = centre[i][1]
    labels.skeleton.delete_node(labels.skeleton.node_names[2])
    labels.skeleton.delete_node(labels.skeleton.node_names[1])
    labels.skeleton.relabel_node(labels.skeleton.node_names[0],"centre")

    labels.save_file(labels, r"labels_centres_"+name+".v001_"+date+".slp")

    return()

#----------------------------------------------------------------------------
# Compute the Radius anf print the values of the average and the median. 
def Radius(labels):
    '''
    Compute the Radius of the ball based on the mean of the radius on all frames. 
    
    labels = the load of a sleap file. There must be three points in the labelled frames. (in the edges of the ball)

    print the mean and the median of all the radius computed from the labelled frames
    '''

    averageRadius = np.mean(np.array(Centres(labels))[:,2])
    medianRadius = np.median(np.array(Centres(labels))[:,2])

    print('The average radius is ', averageRadius, ' .')
    print('The median radius is ', medianRadius, ' .')


    return()

## NOT USEFUL ?
#----------------------------------------------------------------------------
# Add the Centres to the Points 
def AddCentres(labels):

    CentresRadius = np.array(Centres(labels))
    Centres = CentresRadius[:,:-1]
    Points = LabelledPoints(labels)
    Points_Centres = []

    for i in range(len(labels)):
        Points_Centres.append(np.append(Points[i],[Centres[i]],axis = 0))

    return(Points_Centres)

#----------------------------------------------------------------------------
# Get the file of the labelled points. 
labels = slp.load_file(r"/home/matthias/Documents/Sleap/Labels/mmrecorder_balls_labels.v001.slp")

Radius(labels) # 17.4
SaveCentres(labels, "", "230602") # has to be initialized with the right name and date 
  


## Questions : 
# What is I create a new instance and then I merge it with the old one 



## Raw code parts not to forget :
# print(labels[0].instances[0].points[2])
# list(labels[0].instances[0].points[0]) 
# list(Centroids[0])
# labels[0].instances[0].points[0] = Centroids[0]
# list(labels[0].instances[0].points[0]) 
# print(labels[0].instances[0].points[0])
# labels.skeleton.add_node("centre")
# labels.skeleton.delete_node("point3")
# rec_final = np.recarray(final)
# new_skeleton = labels.skeleton.add_node("centre")