import tkinter  as tk
import math
#change colors here
number_color=('dodger blue')
graph_color=('blue2')
text_color=('black')
line_color=('black')
background_color=('white')
def create_graph(window, graph_list, reversed_graph_list, number_color, text_color, graph_color, line_color):
    x =330+(len(graph_list)*5)
    y = 500 * 0.9
    print('python')
    for i in range(len(graph_list)):
      r = (500 - 40) / (len(graph_list) - (len(reversed_graph_list) * 4))
      window.create_rectangle(x, y - 20, x + r, y - reversed_graph_list[i] - 20, fill=graph_color)
      x = (x + r) - 10
      window.create_text(x+20, y-10, text=reversed_graph_list[i], font=("Arial", 7))
      window.create_line(x+7, y, x+7, y-20, fill=line_color)

    hhh=0
    for i in range(21):
      window. create_text(30,(y-20)-hhh,text=(hhh),font=("Arial", 8),fill=number_color)
      window.create_line(20,(y-26)-hhh,40,(y-26)-hhh,fill=line_color)
      hhh=hhh+15
    txt=('percentage')  
    a=list(txt)
    v=225
    for i in range(len(a)):
      window.create_text(10,v,text=a[i],font=("Arial", 8),fill=text_color)
      v=v+10
    window.create_line(0, y-20,500, y-20, fill=line_color)
    window.create_line(0, y,500, y, fill=line_color)
    window.create_line(40,y+10,40,0)
    window.create_line(40,90,460,90)
root = tk.Tk()
root.title("Grapher")
window = tk.Canvas(width=500, height=500, bg="white")
window.pack()
window.create_text(200,50,text='(put title here)', font=('Arial', 16))
window.create_text(200,80,text='(put subtitle here)', font=('Arial', 12))
graph_list = [45,247]
reversed_graph_list = []
for i in range(len(graph_list) - 1, -1, -1):
    reversed_graph_list.append(graph_list[i])

create_graph(window, graph_list, reversed_graph_list, number_color, text_color, graph_color, line_color)


root.mainloop()