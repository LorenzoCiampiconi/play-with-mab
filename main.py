from numpy.random import normal
import PySimpleGUI as sg


if __name__ == '__main__':

    # sg.theme('DarkAmber')  # Add a touch of color
    # All the stuff inside your window.
    layout = [[sg.Text('Qui ci sará la slot')],
              [sg.Button('arm1')],
              [sg.Button('arm2')],
              [sg.Button('arm3')]]

    # Create the Window
    window = sg.Window('Get rich with our MAB', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
            break
        if event == 'pull1':
            samlpe
        if event == 'pull2':
            print('pull2')

    window.close()