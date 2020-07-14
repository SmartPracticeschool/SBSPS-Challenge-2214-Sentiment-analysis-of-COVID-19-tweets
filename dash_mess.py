# set chdir to current dir
import os
import sys
sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))
os.chdir(os.path.realpath(os.path.dirname(__file__)))

import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import sqlite3
import pandas as pd

from collections import Counter
import string
import regex as re
from cache import cache
from config import stop_words
import time
import pickle



conn = sqlite3.connect('twitter.db', check_same_thread=False)

punctuation = [str(i) for i in string.punctuation]



sentiment_colors = {-1:"#EE6055",
                    -0.5:"#FDE74C",
                     0:"#FFE6AC",
                     0.5:"#D0F2DF",
                     1:"#45dd7d",}

#45dd7d
app_colors = {
    'background': '#1a1c23',
    'text': '#FFFFFF',
    'sentiment-plot':'#41EAD4',
    'volume-bar':'#FBFC74',
    'someothercolor':'#FF206E',
}

POS_NEG_NEUT = 0.1

MAX_DF_LENGTH = 100

app = dash.Dash(__name__)
app.layout = html.Div(
    [
  
        
        
        
        html.Div(className= 'row',children=[html.H5('Search:', style={'font-size' : '25px','color':app_colors['text'],'margin-top':'5%'}),
                                                  dcc.Input(id='sentiment_term', value="corona", type='text', style={'font-size' : '20px','color':app_colors['someothercolor'],'width':'18%'}),
                                                  ],
                 style={'margin-top':'10% 0','padding':'5%'}),

        
        
        html.Div(className='row', children=[html.Div(id='related-sentiment', children=html.Button('Loading related terms...', id='related_term_button',style={'margin-top':'30%'})),
                                            html.Div(id='recent-trending',style={"display":"block"})],style={'margin-top':'20%'}),

        html.Div(className='three-col', children=[html.Div(dcc.Graph(id='live-graph',style={"margin-top":"3%"})),
                                            html.Div(dcc.Graph(id='historical-graph',style={"margin-top":"2%"}))],style={'float':'right','margin-top':'15%'}),

        html.Div(className='tweets', children=[html.Div(id="recent-tweets-table"),
                                            html.Div(dcc.Graph(id='sentiment-pie',style={"margin-top":"18%"})),],style={'margin-top':'15%','width':'48%'}),
        html.Img(src=app.get_asset_url('emoji.jpg'),style={'width':'25%','float':'right','margin':'-21%' '15%' '0' '0'}),
        html.Link(
            rel='stylesheet',
            href='/assets/styles.css'
        ),

        html.Link(
            rel='stylesheet',
            href='https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap'
        ),
        
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),
        dcc.Interval(
            id='historical-update',
            interval=10*1000
        ),

        dcc.Interval(
            id='related-update',
            interval=5*1000
        ),

        dcc.Interval(
            id='recent-table-update',
            interval=2*1000
        ),

        dcc.Interval(
            id='sentiment-pie-update',
            interval=10*1000
        ),

    ], style={'background-image': 'url("/assets/headerimg.jpg")' ,'background-size': '100%' '40%','background-repeat': 'no-repeat', 'background-position':'65%' '0%','margin-top':'-5%', 'padding':'0 3%','font-family':'Montserrat'},
)


def df_resample_sizes(df, maxlen=MAX_DF_LENGTH):
    df_len = len(df)
    resample_amt = 100
    vol_df = df.copy()
    vol_df['volume'] = 1

    ms_span = (df.index[-1] - df.index[0]).seconds * 1000
    rs = int(ms_span / maxlen)

    df = df.resample('{}ms'.format(int(rs))).mean()
    df.dropna(inplace=True)

    vol_df = vol_df.resample('{}ms'.format(int(rs))).sum()
    vol_df.dropna(inplace=True)

    df = df.join(vol_df['volume'])

    return df


stop_words.append('')
blacklist_counter = Counter(dict(zip(stop_words, [1000000]*len(stop_words))))


split_regex = re.compile("[ \n"+re.escape("".join(punctuation))+']')

def related_sentiments(df, sentiment_term, how_many=15):
    try:

        related_words = {}

        
        tokens = split_regex.split(' '.join(df['tweet'].values.tolist()).lower())

        
        blacklist_counter_with_term = blacklist_counter.copy()
        blacklist_counter_with_term[sentiment_term] = 1000000
        counts = (Counter(tokens) - blacklist_counter_with_term).most_common(15)

        for term,count in counts:
            try:
                df = pd.read_sql("SELECT sentiment.* FROM  sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 200", conn, params=(term,))
                related_words[term] = [df['sentiment'].mean(), count]
            except Exception as e:
                with open('errors.txt','a') as f:
                    f.write(str(e))
                    f.write('\n')

        return related_words

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


def quick_color(s):
    
    if s >= POS_NEG_NEUT:
        # positive
        return "#45dd7d"
    elif s <= -POS_NEG_NEUT:
        # negative:
        return "#da5657"

    else:
        return '#ffffff'

def generate_table(df, max_rows=10):
    return html.Table(className="responsive-table",
                      children=[
                          html.Thead(
                              html.Tr(
                                  children=[
                                      html.Th(col.title()) for col in df.columns.values],
                                  style={'color':app_colors['text']}
                                  )
                              
                              ),
                          html.Tbody(
                              [
                                  
                              html.Tr(
                                  children=[
                                      html.Td(data) for data in d
                                      ], style={'color':quick_color(d[2]),
                                                'background-color':'#1a1c23'}
                                  )
                               for d in df.values.tolist()])
                          ]
    )


def pos_neg_neutral(col):
    if col >= POS_NEG_NEUT:
        # positive
        return 1
    elif col <= -POS_NEG_NEUT:
        # negative:
        return -1

    else:
        return 0
    
            
@app.callback(Output('recent-tweets-table', 'children'),
              [Input(component_id='sentiment_term', component_property='value'),
			   Input('recent-table-update', 'n_intervals')])        
def update_recent_tweets(sentiment_term,n_intervals):
    if sentiment_term:
        df = pd.read_sql("SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10", conn, params=(sentiment_term+'*',))
    else:
        df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10", conn)

    df['date'] = pd.to_datetime(df['unix'], unit='ms')

    df = df.drop(['unix','id'], axis=1)
    df = df[['date','tweet','sentiment']]

    return generate_table(df, max_rows=10)


@app.callback(Output('sentiment-pie', 'figure'),
              [Input(component_id='sentiment_term', component_property='value'),
			  Input('sentiment-pie-update', 'n_intervals')])
def update_pie_chart(sentiment_term,n_intervals):

    
    for i in range(100):
        sentiment_pie_dict = cache.get('sentiment_shares', sentiment_term)
        if sentiment_pie_dict:
            break
        time.sleep(0.1)

    if not sentiment_pie_dict:
        return None

    labels = ['Positive','Negative']

    try: pos = sentiment_pie_dict[1]
    except: pos = 0

    try: neg = sentiment_pie_dict[-1]
    except: neg = 0

    
    
    values = [pos,neg]
    colors = ['#45dd7d', '#da5657']

    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value', 
                   textfont=dict(size=20, color=app_colors['text']),
                   marker=dict(colors=colors, 
                               line=dict(color=app_colors['background'], width=2)),pull=[0,0.08])

    return {"data":[trace],'layout' : go.Layout(
                                                  title='Positive vs Negative sentiment for "{}" (longer-term)'.format(sentiment_term),
                                                  font={'color':app_colors['text']},
                                                  plot_bgcolor = app_colors['background'],
                                                  paper_bgcolor = app_colors['background'],
                                                  showlegend=True,
                                                  margin=dict(l=0, r=10))}




@app.callback(Output('live-graph', 'figure'),
              [Input('sentiment_term','value'), Input('graph-update', 'n_intervals')])
def update_graph_scatter(sentiment_term,n_intervals):
    try:
        if sentiment_term:
            df = pd.read_sql("SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 1000", conn, params=(sentiment_term+'*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 1000", conn)
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        init_length = len(df)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df = df_resample_sizes(df)
        X = df.index
        Y = df.sentiment_smoothed.values
        Y2 = df.volume.values
        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Sentiment',
                mode= 'lines',
                yaxis='y2',
                line = dict(color = (app_colors['sentiment-plot']),
                            width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color=app_colors['volume-bar']),
                )

        return {'data': [data,data2],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                          yaxis=dict(range=[min(Y2),max(Y2*4)], title='Volume', side='right'),
                                                          yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='sentiment'),
                                                          title='Live sentiment for: "{}"'.format(sentiment_term),
                                                          font={'color':app_colors['text']},
                                                          plot_bgcolor = app_colors['background'],
                                                          paper_bgcolor = app_colors['background'],
                                                          showlegend=False)}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')
		
		
@app.callback(Output('historical-graph', 'figure'),
              [Input('sentiment_term','value'),Input('historical-update', 'n_intervals')])
def update_hist_graph_scatter(sentiment_term,n_intervals):
    try:
        if sentiment_term:
            df = pd.read_sql("SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10000", conn, params=(sentiment_term+'*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10000", conn)
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        
        cache.set('related_terms', sentiment_term, related_sentiments(df, sentiment_term), 120)

        
        init_length = len(df)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df.dropna(inplace=True)
        df = df_resample_sizes(df,maxlen=500)
        X = df.index
        Y = df.sentiment_smoothed.values
        Y2 = df.volume.values

        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Sentiment',
                mode= 'lines',
                yaxis='y2',
                line = dict(color = (app_colors['sentiment-plot']),
                            width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color=app_colors['volume-bar']),
                )

        df['sentiment_shares'] = list(map(pos_neg_neutral, df['sentiment']))

        
        cache.set('sentiment_shares', sentiment_term, dict(df['sentiment_shares'].value_counts()), 120)

        return {'data': [data,data2],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]), # add type='category to remove gaps'
                                                          yaxis=dict(range=[min(Y2),max(Y2*4)], title='Volume', side='right'),
                                                          yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='sentiment'),
                                                          title='Longer-term sentiment for: "{}"'.format(sentiment_term),
                                                          font={'color':app_colors['text']},
                                                          plot_bgcolor = app_colors['background'],
                                                          paper_bgcolor = app_colors['background'],
                                                          showlegend=False)}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')



max_size_change = .4

def generate_size(value, smin, smax):
    size_change = round((( (value-smin) /smax)*2) - 1,2)
    final_size = (size_change*max_size_change) + 1
    return final_size*120




@app.callback(Output('related-sentiment', 'children'),
              [Input(component_id='sentiment_term', component_property='value'),Input('related-update', 'n_intervals')])
def update_related_terms(sentiment_term,n_intervals):
    try:

        # get data from cache
        for i in range(100):
            related_terms = cache.get('related_terms', sentiment_term) # term: {mean sentiment, count}
            if related_terms:
                break
            time.sleep(0.1)

        if not related_terms:
            return None

        buttons = [html.Button('{}({})'.format(term, related_terms[term][1]), id='related_term_button', value=term, className='btn', type='submit', style={'background-color':'#4CBFE1',
                                                                                                                                                           'margin-right':'5px',
                                                                                                                                                           'margin-top':'5px'}) for term in related_terms]
        
        

        sizes = [related_terms[term][1] for term in related_terms]
        smin = min(sizes)
        smax = max(sizes) - smin  

        buttons = [html.H5('Terms related to "{}" '.format(sentiment_term),className='button',style={'font-size' : '25px','color':app_colors['text'],'margin-right':'15px'})]+[html.Span(term, style={'color':sentiment_colors[round(related_terms[term][0]*2)/2],
                                                              'margin-right':'15px',
                                                              'margin-top':'5%',
                                                              'font-size' : '25px'}) for term in related_terms]


        return buttons
        

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')




@app.callback(Output('recent-trending', 'children'),
              [Input(component_id='sentiment_term', component_property='value'),Input('related-update', 'n_intervals')])
def update_recent_trending(sentiment_term,n_intervals):
    try:
        query = """
                SELECT
                        value
                FROM
                        misc
                WHERE
                        key = 'trending'
        """

        c = conn.cursor()

        result = c.execute(query).fetchone()

        related_terms = pickle.loads(result[0])




        

        sizes = [related_terms[term][1] for term in related_terms]
        smin = min(sizes)
        smax = max(sizes) - smin  

        buttons = [html.H5('Recently Trending Terms ', style={'color':app_colors['text'],'font-size' : '25px','margin':'5% 15px 0 0'},className='button')]+[html.Span(term, style={'color':sentiment_colors[round(related_terms[term][0]*2)/2],
                                                              'margin-right':'15px',
                                                              'margin-top':'15px',
                                                              'font-size' : '25px'}) for term in related_terms]


        return buttons
        

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


            

server = app.server
dev_server = app.run_server

#if __name__ == '__main__':
#    app.run_server(debug=True)
