from numpy.random import normal
import PySimpleGUI as sg
from PySimpleGUI import Column
from PIL import Image
import io

if __name__ == '__main__':

    # sg.theme('DarkAmber')  # Add a touch of color
    # All the stuff inside your window.



    # window["-IMAGE-"].update(data=bio.getvalue())

    col1 = [[sg.Image(data=bio.getvalue())], [sg.Button('arm1', size=(5, 1.3))], [sg.Text('', key="text1")]]
    col2 = [[sg.Image(data=bio.getvalue())], [sg.Button('arm2', size=(5, 1.3))], [sg.Text('', key="text2")]]
    col3 = [[sg.Image(data=bio.getvalue())], [sg.Button('arm3', size=(5, 1.3))], [sg.Text('', key="text3")]]
    layout = [[sg.Column(col1, element_justification='c'), sg.Column(col2, element_justification='c')
        , sg.Column(col3, element_justification='c')]]

    # Create the Window
    window = sg.Window('Get rich with our MAB', layout, size=(1250, 800))

    test = ["0.5","0.4","0.3"]
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
            break
        if event == 'arm1':
            window['text1'].update('\n'.join(test))
        if event == 'arm2':
            window['text2'].update('new text2')

    window.close()