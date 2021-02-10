from app import app 
from app import server 
import dash_html_components as html 
import dash_core_components as dcc 

app.layout = html.Div([html.H1('Dash Demo Graph', 
                              style={
                                  'textAlign': 'center'
                                  "background": 'yellow'}), 
                      dcc.Graph( 
                          id='graph-1', 
                          figure={ 
                              'data': [ 
                                  {'x': [1,2,3,4,5,6,7], 'y':[10,20,30,40,50,60,70], 'type': 'line', 'name': 'value'},
                                  {'x': [1, 2, 3, 4, 5, 6, 7], 'y': [12, 22, 36, 44, 49, 58, 73], 'type': 'line', 'name':'value2'}
                              ], 
                              'layout': { 
                                  'title': 'Simply Line Graph', 
                                   }
                               }
                          ), 
                          ], style = { 
                              "background": "#00080"}
                          ) 
if __name__ == '__main__': 
    app.run_server(debug=True)