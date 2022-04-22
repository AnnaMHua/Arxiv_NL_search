from nltk import tokenize
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import networkx as nx

import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.models.fasttext import FastText
from rank_bm25 import BM25Okapi
import nmslib
import time

import utils

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# ------------- indexing -----------

index_title = nmslib.init(method='hnsw', space='cosinesimil')
index_author = nmslib.init(method='hnsw', space='cosinesimil')
index_categories = nmslib.init(method='hnsw', space='cosinesimil')

index_title.loadIndex("index_title.bin")
index_author.loadIndex("index_author.bin")
index_categories.loadIndex("index_categories.bin")

# reading the arxiv data
df = pd.read_csv("data/arxiv_smaller.csv")

# ------------- Define layout for the app ----------------

app.layout = html.Div([
    dcc.Tabs(id='tabs-nav', value='tab-1', children=[
        dcc.Tab(label='Search engine', value='tab-1'),
        dcc.Tab(label='Data', value='tab-2'),
        dcc.Tab(label='About', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-nav', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                dcc.Markdown('''
                    ## WahooGo Search
                    Search for ArXiv Papers
                
                    '''
                ),
            ], style={'padding': 10, 'text-align': 'center'}),
            html.Div([
                dcc.Markdown('''
                    **Search terms:**
                    '''),
                dcc.Input(id='search-terms-input', type='text', value='computer science papers'),
                dcc.Markdown('''
                    **Show this many rows:**
                    '''
                ),
                dcc.Input(
                        id="range-limit", type="number", placeholder="input with range",
                        min=1, max=60, step=1, value=10,
                ),
                dcc.Markdown('''
                    **Starting with row:**
                    '''
                ),
                dcc.Input(
                    id="range-offset", type="number", placeholder="input with range",
                    min=1, max=60, step=1, value=1,
                ),
            ], style={'columnCount': 4}),    
            html.Div([
                html.Button(id='submit-button-state', n_clicks=0, children='Run'),
            ], style={'padding': 20, 'text-align': 'center'}),
            html.Br(),
            dcc.Markdown('''
                ###### Notes:
                - Stopwords ("to", "the", "a", etc.) will be ignored. "to be or not to be" will thus return nothing.
                - Searching will sometimes take up to 1 minute depending on the query
                '''
            ),
            html.Div(id='search-terms-active'),
            html.Br(),
            html.Div(
                id = 'tableDiv',
                className = 'tableDiv'
            ),
            dcc.Loading(
                        id="loading-1",
                        type="default",
                        children=html.Div(id="tableDiv"))
        ], style={'width': '70%', 'margin': 'auto'})
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                dcc.Markdown('''
                    ## The data
                    '''
                )
            ], style={'padding': 10, 'text-align': 'center'}),
            html.Div([
                dcc.Markdown('''
                    
                    #### Source
                    
                    Cornell University has created a machine readable arXiv dataset with 1.7 million articles. The dataset is available
                    for free on Kaggle.                 
                
                    '''
                ),
            ]),
        ], style={'width': '70%', 'margin': 'auto'})
    elif tab == 'tab-3':
        return html.Div(
        [
            html.Div([
                dcc.Markdown('''
                    ## About
                    ''')
        ], style={'padding': 10, 'text-align': 'center'}),
        html.Div([
            dcc.Markdown('''
                This search engine was made for our Information Retrieval class at the University of Virginia. 
                #### GDPR, PDPA etc.
                We urrently track nothing that we know of except your input when you press submit, and the timestamp thereof. 
                        
                '''
            )
        ], style={'padding': 10}),
        dcc.Markdown('''
        '''),
        ], style={'width': '70%', 'margin': 'auto'})


# ------------- Make the app interactive -----------------
@app.callback(
    Output('tableDiv', 'children'),
    Input('submit-button-state', 'n_clicks'), 
    State('search-terms-input', 'value'),
    State('range-limit', 'value'),
    State('range-offset', 'value')
)
def update_table(n_clicks, search_terms, limit, offset):             
    input = search_terms

    tokenized_input = utils.get_post(input)
    dates = utils.get_dates(tokenized_input)
    authors = utils.get_authors(tokenized_input)
    text, important = utils.time_important(utils.get_input(tokenized_input))
    text = text.split()
    
    ft_model_title = FastText.load('_fasttext_title.model')
    ft_model_author = FastText.load('_fasttext_author.model')
    ft_model_categories = FastText.load('_fasttext_categories.model')
    
    query_title = [ft_model_title.wv[vec] for vec in text]
    query_title = np.mean(query_title,axis=0)

    query_author = [ft_model_author.wv[vec] for vec in authors]
    query_author = np.mean(query_author,axis=0)

    query_categories = [ft_model_categories.wv[vec] for vec in text]
    query_categories = np.mean(query_categories,axis=0)

    ids_title, distances_title = index_title.knnQuery(query_title, k=10)
    titles_df = df.iloc[ids_title]
    titles_df['distances'] = pd.DataFrame(pd.Series(distances_title)).set_index(titles_df.index)
    titles_df.reset_index(inplace=True)
    results = titles_df
    
    if not np.isnan(query_author).any(): 
        ids_author, distances_author = index_author.knnQuery(query_author, k=10)
        authors_df = df.iloc[ids_author]
        authors_df['distances'] = pd.DataFrame(pd.Series(distances_author)).set_index(authors_df.index)
        authors_df.reset_index(inplace=True)
        results = pd.concat([authors_df, titles_df])

    ids_categories, distances_categories = index_categories.knnQuery(query_categories, k=10)
    categories_df = df.iloc[ids_categories]
    categories_df['distances'] = pd.DataFrame(pd.Series(distances_categories)).set_index(categories_df.index)
    categories_df.reset_index(inplace=True)
    results = pd.concat([results, categories_df])
    results['distances'] = results['distances'].apply(lambda x: round(x, 2))

    if important: 
        dict_results = results.sort_values(by=['update_date','distances'])[:10].to_dict("records")
    else: 
        dict_results = results.sort_values(by=['distances'])[:10].to_dict('records')
    
    #Create Table
    tbl = dash_table.DataTable(
        id = 'table',
        style_cell={
                'whiteSpace': 'normal',
                'height': 'auto',
                'textAlign': 'left',
                'font-family':'sans-serif', 
                'overflow': 'hidden', 
                'textOverflow': 'ellipsis', 
                'maxWidth': 0, 
            },
        data=dict_results,
        columns=[
            {'name': 'Author', 'id':'authors', 'type':'text'}, 
            {'name': 'Title', 'id':'title', 'type':'text'}, 
            {'name': 'Abstract', 'id':'abstract', 'type':'string'}, 
            {'name': 'Distances', 'id':'distances', 'type':'numeric'}, 
        ],
        tooltip_delay=0, 
        tooltip_duration=None, 
        markdown_options={'link_target': '_blank'},
        filter_action = 'native',
        sort_action = 'native',
        sort_mode = 'multi',
        export_format="csv",
        
    )
    return html.Div([tbl])

@app.callback(Output('search-terms-active', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('search-terms-input', 'value'),
              State('range-limit', 'value'),
              State('range-offset', 'value'))
def update_output(n_clicks, input, range_limit, range_offset):
    return '''
        Current selection: "{}" starting with result no. {} for {} rows. Matched words are highlighted. May take up to 1 minute to load.
    '''.format(input, range_offset, range_limit)

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
    