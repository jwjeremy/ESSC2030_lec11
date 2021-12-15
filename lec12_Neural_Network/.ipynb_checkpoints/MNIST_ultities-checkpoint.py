from tkinter import *
import time
from PIL import Image, ImageDraw
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# This function takes 2 arguments, 
#   - an optional keras model,
#   - Canvas Width allows user to set the pixel height and width of the drawing box
#
#  The function returns a 28*28 matrix of  pixel values made from the canvas drawing
#  if a keras model is also provided then the function will plot prediction results too

def DrawMyOwnNumbers(Kerasmodel=None,CanvasWidth=200):
    DrawnOutMatrix=0
    width=CanvasWidth
    white = (255, 255, 255)
    black = (0, 0, 0)

    b1 = "up"
    xold, yold = None, None    

    def b1down(event):
        nonlocal b1
        b1 = "down"           

    def b1up(event):
        nonlocal b1, xold, yold
        b1 = "up"
        xold = None
        yold = None
    def motion(event):
            if b1 == "down":
                nonlocal xold, yold
                if xold is not None and yold is not None:
                    event.widget.create_line(xold,yold,event.x,event.y)
                    draw.line([xold,yold,event.x,event.y], black)
                xold = event.x
                yold = event.y
    root = Tk()

    drawing_area = Canvas(root,width=width,height=width)
    image1 = Image.new("RGB", (width, width), white)
    draw = ImageDraw.Draw(image1)
    drawing_area.pack()
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)
    
    def handle_click():
        nonlocal DrawnOutMatrix
        nonlocal root
        root.destroy()
        #%matplotlib inline


        filename = "4my_drawing.bmp"
        image1.save(filename)
        II=np.asarray(mpimg.imread(filename))
        II=np.apply_along_axis(np.min,2,II)
        

        data=np.asarray([[i,j] for i in range(width) for j in range(width) if II[j,i]==0]).astype('float64')
        datax=data[:,0]
        datay=data[:,1]
        
        # centre and scale the pixels
        datax -= np.mean(datax)
        datay -= np.mean(datay)
        scaling = np.max(np.abs([datax.min(),datax.max(),datay.min(),datay.max()]))
        datax *= 1/scaling
        datay *= 1/scaling
      
        posn = np.linspace(-1.2,1.2, 28)
        sx = -0.5*(1/0.06)**2
        def kde(x,y):
            return(np.min([np.sum(np.exp(sx*((x-datax)**2+(y-datay)**2))),5]))

        output=np.asarray([[kde(x,y) for x in posn] for y in posn])
        output=output/output.max()
        output1=np.asarray([[[ [output[j,i]] for i in np.arange(28)] for j in np.arange(28)]])*256
        DrawnOutMatrix=output1[0,:,:,0]
        
        if Kerasmodel:
            p= Kerasmodel.predict_proba(output1,verbose=1==2)[0]

            f, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,5))
            ax1.imshow(255-II,cmap='Greys')

            def PlotMnist(inn,ax):
                ax.imshow(inn[0,:,:,0],cmap='Greys')

            PlotMnist(output1,ax2)
            ax3.bar(np.arange(10),p,0.8,color='g')


    Button(root, text='classify!', command=handle_click).pack()
    root.mainloop()
    return DrawnOutMatrix
