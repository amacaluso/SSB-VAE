import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Graphics Libraries Import
from IPython.display import HTML  #For Gui
from tkinter import *             #For Graphic
from tkinter.ttk import Combobox  #Combobox
from optparse import OptionParser

op = OptionParser()
op.add_option("-c", "--nbits", type=int, default=16, help="number of bits")
op.add_option("-t", "--table", type="string", help="table from which generates new values")

(opts, args) = op.parse_args()
nbits = opts.nbits
#Data structures for comboboxes
dataset=("20News", "CIFAR-10", "Snippets", "TMC")
level=("0.1", "0.2", "0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0")


#Function 1 - Create dataframes for each dataset
def load_dataset():
    dataset = pd.read_csv("ResultsPostProcessing/"+opts.table)
    return dataset

#Function 2 - Reads the selected dataset via combobox e create the name of the corresponding dataframe
#Call by Combobox 1
def readDataset():
    dataset = cbChoiceDataset.get()
    if dataset == "CIFAR-10":
        dataset = "CIFAR"
    datasetName = "df_" + dataset
    return datasetName

#Function 3 - Reads the selected level via combobox
#Call by Combobox 1
def readLevel():
    levelName = cbChoiceLevel.get()
    return levelName

#Function 4 - View correct data
#Call by RadioButton 1
def radioParam():
    if vChoiceParam.get() == 1:
        lblValueAlpha.place_forget()
        lblValueBeta.place(x=70, y=200)
        lblValueLambda.place(x=70, y=230)
    elif vChoiceParam.get() == 2:
        lblValueBeta.place_forget()
        lblValueAlpha.place(x=70, y=200)
        lblValueLambda.place(x=70, y=230)
    else:
        lblValueLambda.place_forget()
        lblValueAlpha.place(x=70, y=200)
        lblValueBeta.place(x=70, y=230)
        
#Function 5 - Collects and displays the values to choose from on the screen
#Call by Button 1
def showValues():
    name = readDataset()
    df = eval(name)
    lev = readLevel()
    end = int(df.shape[0])
    start = end - 270
    df = df.iloc[start:end]
    df_level = df[df["level"] == float(lev)]
    if vChoiceParam.get() == 1:
        betaValues = df_level["beta"].unique()
        lambdaValues = df_level["lambda"].unique()

        t1Value1.delete(0, 'end')
        t1Value1.insert(END, str(betaValues[0]))
        t1Value2.delete(0, 'end')
        t1Value2.insert(END, str(betaValues[1]))
        t1Value3.delete(0, 'end')
        t1Value3.insert(END, str(betaValues[2]))

        t2Value1.delete(0, 'end')
        t2Value1.insert(END, str(lambdaValues[0]))
        t2Value2.delete(0, 'end')
        t2Value2.insert(END, str(lambdaValues[1]))
        t2Value3.delete(0, 'end')
        t2Value3.insert(END, str(lambdaValues[2]))

    elif vChoiceParam.get() == 2:
        alphaValues = df_level["alpha"].unique()
        lambdaValues = df_level["lambda"].unique()       

        t1Value1.delete(0, 'end')
        t1Value1.insert(END, str(alphaValues[0]))
        t1Value2.delete(0, 'end')
        t1Value2.insert(END, str(alphaValues[1]))
        t1Value3.delete(0, 'end')
        t1Value3.insert(END, str(alphaValues[2]))

        t2Value1.delete(0, 'end')
        t2Value1.insert(END, str(lambdaValues[0]))
        t2Value2.delete(0, 'end')
        t2Value2.insert(END, str(lambdaValues[1]))
        t2Value3.delete(0, 'end')
        t2Value3.insert(END, str(lambdaValues[2]))

    else:
        alphaValues = df_level["alpha"].unique()
        betaValues = df_level["beta"].unique()

        t1Value1.delete(0, 'end')
        t1Value1.insert(END, str(alphaValues[0]))
        t1Value2.delete(0, 'end')
        t1Value2.insert(END, str(alphaValues[1]))
        t1Value3.delete(0, 'end')
        t1Value3.insert(END, str(alphaValues[2]))

        t2Value1.delete(0, 'end')
        t2Value1.insert(END, str(betaValues[0]))
        t2Value2.delete(0, 'end')
        t2Value2.insert(END, str(betaValues[1]))
        t2Value3.delete(0, 'end')
        t2Value3.insert(END, str(betaValues[2]))

#Function 6 - Returns the index of the chosen value
#Call by RadioButton 2 - 1 row
def radioValueParam1():
    if vChoiceValueParam1.get() == 1:
        return 0
    elif vChoiceValueParam1.get() == 2:
        return 1
    else:
        return 2

#Function 7 - Returns the index of the chosen value
#Call by RadioButton 2 - 2 row 
def radioValueParam2():
    if vChoiceValueParam2.get() == 1:
        return 0
    elif vChoiceValueParam2.get() == 2:
        return 1
    else:
        return 2

#Function 8 - Create the Plot
#Call by Button 2
def createPlot():
    name = readDataset()
    df = eval(name)
    lev = readLevel()
    df_level = df[df["level"] == float(lev)]
    if vChoiceParam.get() == 1:
        ind1 = radioValueParam1()
        ind2 = radioValueParam2()
        betaValues = df_level["beta"].unique()
        lambdaValues = df_level["lambda"].unique()
        betaV = betaValues[ind1]
        lambdaV = lambdaValues[ind2]
        dfFinal = df_level[(df_level["beta"] == float(betaV)) & (df_level["lambda"] == float(lambdaV))]
        finalPlot = dfFinal.plot(x = 'alpha', y = 'p@100', kind = 'scatter', color = 'black')
        finalPlot.set_title('Precision with Alpha')
        finalPlot.spines['top'].set_visible(False)
        finalPlot.spines['right'].set_visible(False)
        finalPlot.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        for i in range(0,dfFinal.shape[0]):
            finalPlot.annotate((dfFinal.iloc[i,3],dfFinal.iloc[i,4],dfFinal.iloc[i,5]), (dfFinal.iloc[i,3],dfFinal.iloc[i,6]))
        plt.show()
    elif vChoiceParam.get() == 2:
        ind1 = radioValueParam1()
        ind2 = radioValueParam2()
        alphaValues = df_level["alpha"].unique()
        lambdaValues = df_level["lambda"].unique()
        alphaV = alphaValues[ind1]
        lambdaV = lambdaValues[ind2]
        dfFinal = df_level[(df_level["alpha"] == float(alphaV)) & (df_level["lambda"] == float(lambdaV))]
        finalPlot = dfFinal.plot(x = 'beta', y = 'p@100', kind = 'scatter', color = 'black')
        finalPlot.set_title('Precision with Beta')
        finalPlot.spines['top'].set_visible(False)
        finalPlot.spines['right'].set_visible(False)
        finalPlot.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        for i in range(0,dfFinal.shape[0]):
            finalPlot.annotate((dfFinal.iloc[i,3],dfFinal.iloc[i,4],dfFinal.iloc[i,5]), (dfFinal.iloc[i,4],dfFinal.iloc[i,6]))
        plt.show()
    else:
        ind1 = radioValueParam1()
        ind2 = radioValueParam2()
        alphaValues = df_level["alpha"].unique()
        betaValues = df_level["beta"].unique()
        alphaV = alphaValues[ind1]
        betaV = betaValues[ind2]
        dfFinal = df_level[(df_level["alpha"] == float(alphaV)) & (df_level["beta"] == float(betaV))]
        finalPlot = dfFinal.plot(x = 'lambda', y = 'p@100', kind = 'scatter', color = 'black')
        finalPlot.set_title('Precision with Lambda')
        finalPlot.spines['top'].set_visible(False)
        finalPlot.spines['right'].set_visible(False)
        finalPlot.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        for i in range(0,dfFinal.shape[0]):
            finalPlot.annotate((dfFinal.iloc[i,3],dfFinal.iloc[i,4],dfFinal.iloc[i,5]), (dfFinal.iloc[i,5],dfFinal.iloc[i,6]))
        plt.show()


graphicWindow = Tk()                            #Creation of the window that contains the various widgets
graphicWindow.title('Charts')                   #Window Title
graphicWindow.geometry("1500x800+10+20")        #Window Size
#stateWindow.configure(background = 'white')    #Wondow Colour


#Create dataframes for each dataset
data = load_dataset()   #Call Function 1

df_20News=data[data["dataset"] == "20News"]
df_TMC=data[data["dataset"] == "TMC"]
df_Snippets=data[data["dataset"] == "Snippets"]
df_CIFAR=data[data["dataset"] == "CIFAR-10"]

#Label 1 - Title of The Window
lblWindowTitle = Label(graphicWindow, text="Dynamic Charts", fg='red', font=("Arial", 16))   
lblWindowTitle.place(x=700, y=10)

#Label 2 & Combobox 1 - Choice of Dataset
lblChoiceDataset = Label(graphicWindow, text="Choice the Dataset: ", fg='black', font=("Arial", 14))
lblChoiceDataset.place(x=50,y=50)

cbChoiceDataset = Combobox(graphicWindow, values=dataset) 
cbChoiceDataset.place(x=230,y=53)
cbChoiceDataset.bind("<<ComboboxSelected>>", lambda e: readDataset())   #Call Function of Reading - Function 2

#Label 3 & Combobox 2 - Choice of Level
lblChoiceLevel = Label(graphicWindow, text="Choice the Level: ", fg='black', font=("Arial", 14))
lblChoiceLevel.place(x=50,y=80)

cbChoiceLevel = Combobox(graphicWindow, values=level) 
cbChoiceLevel.place(x=230,y=83)
cbChoiceLevel.bind("<<ComboboxSelected>>", lambda e: readLevel())   #Call Function of Reading - Function 3

#Label 3 & RadioButton 1 - Choice of Hyperparameters to Displayed
lblChoiceHyperparameter = Label(graphicWindow, text="Choose the hyperparameter to view: ", fg='black', font=("Arial", 14))
lblChoiceHyperparameter.place(x=50,y=110)

#Radiobutton 1 - Allows you to choose which Parameter to display
vChoiceParam = IntVar() 
vChoiceParam.set(1)
rAlpha = Radiobutton(graphicWindow, text="Alpha", variable=vChoiceParam,value=1, command = lambda : radioParam(), fg='blue', font=("Arial", 12))   #Call the display function - Function 4
rBeta = Radiobutton(graphicWindow, text="Beta", variable=vChoiceParam,value=2, command = lambda : radioParam(), fg='blue', font=("Arial", 12))
rLambda = Radiobutton(graphicWindow, text="Lambda", variable=vChoiceParam,value=3, command = lambda : radioParam(), fg='blue', font=("Arial", 12))
rAlpha.place(x=70, y=140)
rBeta.place(x=140,y=140)
rLambda.place(x=210,y=140)

#Button 1 - It displays on the screen the values to choose from for the parameters to be set
btnShowValues = Button(graphicWindow, text="Confirmation", fg='Red', font=("Arial", 10))
btnShowValues.place(x=370, y=127)
btnShowValues.bind('<Button-1>', lambda e: showValues()) #Collects and displays the values to choose from on the screen - Function 5 

#Label 4 - Choice the value of the other Hyperparameters 
lblChoiceValueHyperparameter = Label(graphicWindow, text="Choose the value of the other hyperparameter: ", fg='black', font=("Arial", 14))
lblChoiceValueHyperparameter.place(x=50,y=170)

#Label 5 - Choice the value of Alpha
lblValueAlpha = Label(graphicWindow, text="Choose the value with which to fix Alpha: ", fg='black', font=("Arial", 12))
lblValueAlpha.place_forget()

#Label 6 - Choice the value of Beta
lblValueBeta = Label(graphicWindow, text="Choose the value with which to fix Beta: ", fg='black', font=("Arial", 12))
lblValueBeta.place(x=70, y=200)

#Label 7 - Choice the value of Lambda
lblValueLambda = Label(graphicWindow, text="Choose the value with which to fix Lambda: ", fg='black', font=("Arial", 12))
lblValueLambda.place(x=70, y=230)

#Entry 1 & RadioButton 2- They contain values to choose from
t1Value1 = Entry(graphicWindow, width="10")
t1Value1.place(x=380, y=203)
t1Value2 = Entry(graphicWindow, width="10")
t1Value2.place(x=505, y=203)
t1Value3 = Entry(graphicWindow, width="10")
t1Value3.place(x=625, y=203)

t2Value1 = Entry(graphicWindow, width="10")
t2Value1.place(x=380, y=233)
t2Value2 = Entry(graphicWindow, width="10")
t2Value2.place(x=505, y=233)
t2Value3 = Entry(graphicWindow, width="10")
t2Value3.place(x=625, y=233)

vChoiceValueParam1 = IntVar() 
vChoiceValueParam1.set(1)
r1Value1 = Radiobutton(graphicWindow, text="", variable=vChoiceValueParam1,value=1, command = lambda : radioValueParam1(), fg='blue', font=("Arial", 12))   #Choice the fixed values of first row - Function 6
r1Value2 = Radiobutton(graphicWindow, text="", variable=vChoiceValueParam1,value=2, command = lambda : radioValueParam1(), fg='blue', font=("Arial", 12))
r1Value3 = Radiobutton(graphicWindow, text="", variable=vChoiceValueParam1,value=3, command = lambda : radioValueParam1(), fg='blue', font=("Arial", 12))
r1Value1.place(x=450, y=201)
r1Value2.place(x=575,y=201)
r1Value3.place(x=685,y=201)

vChoiceValueParam2 = IntVar() 
vChoiceValueParam2.set(1)
r2Value1 = Radiobutton(graphicWindow, text="", variable=vChoiceValueParam2,value=1, command = lambda : radioValueParam2(), fg='blue', font=("Arial", 12))   #Choice the fixed values of 2nd row - Function 7
r2Value2 = Radiobutton(graphicWindow, text="", variable=vChoiceValueParam2,value=2, command = lambda : radioValueParam2(), fg='blue', font=("Arial", 12))
r2Value3 = Radiobutton(graphicWindow, text="", variable=vChoiceValueParam2,value=3, command = lambda : radioValueParam2(), fg='blue', font=("Arial", 12))
r2Value1.place(x=450, y=231)
r2Value2.place(x=575,y=231)
r2Value3.place(x=685,y=231)

#Button 2 - Button to create the graph and confirm all choices
btnConfirmAll = Button(graphicWindow, text="Confirmation", fg='Red', font=("Arial", 10))
btnConfirmAll.place(x=400, y=270)
btnConfirmAll.bind('<Button-1>', lambda e: createPlot()) #Collects and displays the values to choose from on the screen - Function 8

graphicWindow.mainloop()  #Window's Execution