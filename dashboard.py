import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import scipy
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_csv('new_final_dataset.csv')

corr_matrix = df[['SAT Critical Reading 75th percentile score', 'SAT Math 75th percentile score',
                      'ACT Composite 75th percentile score', 'Percent admitted - total',
                      'Percent of freshmen receiving any financial aid', 'Percent of total enrollment that are women',
                      'Graduation rate - Bachelor degree within 4 years',
                      'Percent of total enrollment that are White', 'County Poverty Rate 18 to 34',
                      'Percent of total enrollment that are Black or African American']].corr()


def pcd_dataframe_selecter(dataframe, restyleData):

    print("inside method")
    pcd_df = dataframe
    print(restyleData[0])

    if 'dimensions[0].constraintrange' in restyleData[0]:
        print('constraint placed on 0')
        print(restyleData[0]['dimensions[0].constraintrange'][0])

        lower_sat = restyleData[0]['dimensions[0].constraintrange'][0][0]
        higher_sat = restyleData[0]['dimensions[0].constraintrange'][0][1]
        pcd_df = pcd_df.loc[(pcd_df['SAT Critical Reading 75th percentile score'] >= lower_sat) &
                            (pcd_df['SAT Critical Reading 75th percentile score'] <= higher_sat)]

    if 'dimensions[1].constraintrange' in restyleData[0]:
        print('constraint placed on 1')
        lower_sat = restyleData[0]['dimensions[1].constraintrange'][0][0]
        higher_sat = restyleData[0]['dimensions[1].constraintrange'][0][1]
        pcd_df = pcd_df.loc[(pcd_df['SAT Math 75th percentile score'] >= lower_sat) &
                            (pcd_df['SAT Math 75th percentile score'] <= higher_sat)]

    if 'dimensions[2].constraintrange' in restyleData[0]:
        print('constraint placed on 2')
        lower_act = restyleData[0]['dimensions[2].constraintrange'][0][0]
        higher_act = restyleData[0]['dimensions[2].constraintrange'][0][1]
        pcd_df = pcd_df.loc[(pcd_df['ACT Composite 75th percentile score'] >= lower_act) &
                            (pcd_df['ACT Composite 75th percentile score'] <= higher_act)]

    if 'dimensions[3].constraintrange' in restyleData[0]:
        print('constraint placed on 3')
        lower_grad = restyleData[0]['dimensions[3].constraintrange'][0][0]
        higher_grad = restyleData[0]['dimensions[3].constraintrange'][0][1]
        pcd_df = pcd_df.loc[(pcd_df['Graduation rate - Bachelor degree within 4 years'] >= lower_grad) &
                            (pcd_df['Graduation rate - Bachelor degree within 4 years'] <= higher_grad)]

    if 'dimensions[4].constraintrange' in restyleData[0]:
        print('constraint placed on 4')
        lower_blk = restyleData[0]['dimensions[4].constraintrange'][0][0]
        higher_blk = restyleData[0]['dimensions[4].constraintrange'][0][1]
        pcd_df = pcd_df.loc[(pcd_df['Percent of total enrollment that are Black or African American'] >= lower_blk) &
                            (pcd_df['Percent of total enrollment that are Black or African American'] <= higher_blk)]

    if 'dimensions[5].constraintrange' in restyleData[0]:
        print('constraint placed on 5')
        lower_wht = restyleData[0]['dimensions[5].constraintrange'][0][0]
        higher_wht = restyleData[0]['dimensions[5].constraintrange'][0][1]
        pcd_df = pcd_df.loc[(pcd_df['Percent of total enrollment that are White'] >= lower_wht) &
                            (pcd_df['Percent of total enrollment that are White'] <= higher_wht)]

    if 'dimensions[6].constraintrange' in restyleData[0]:
        print('constraint placed on 6')
        lower_pct = restyleData[0]['dimensions[6].constraintrange'][0][0]
        higher_pct = restyleData[0]['dimensions[6].constraintrange'][0][1]
        pcd_df = pcd_df.loc[(pcd_df['Percent admitted - total'] >= lower_pct) &
                            (pcd_df['Percent admitted - total'] <= higher_pct)]

    if 'dimensions[7].constraintrange' in restyleData[0]:
        print('constraint placed on 7')
        lower_aid = restyleData[0]['dimensions[7].constraintrange'][0][0]
        higher_aid = restyleData[0]['dimensions[7].constraintrange'][0][1]
        pcd_df = pcd_df.loc[(pcd_df['Percent of freshmen receiving any financial aid'] >= lower_aid) &
                            (pcd_df['Percent of freshmen receiving any financial aid'] <= higher_aid)]

    if 'dimensions[8].constraintrange' in restyleData[0]:
        print('constraint placed on 8')
        lower_women = restyleData[0]['dimensions[8].constraintrange'][0][0]
        higher_women = restyleData[0]['dimensions[8].constraintrange'][0][1]
        pcd_df = pcd_df.loc[(pcd_df['Percent of total enrollment that are women'] >= lower_women) &
                            (pcd_df['Percent of total enrollment that are women'] <= higher_women)]

    return pcd_df

@app.callback(
    Output(component_id='parallel_coordinates_display', component_property='figure'),
    Input(component_id='act_pie_chart', component_property='clickData'),
    Input(component_id='scatter_white_vs_admitted', component_property='selectedData'),
    Input(component_id='parallel_coordinates_display', component_property='restyleData'),
    Input(component_id='graduation_histogram', component_property='clickData'),
    Input(component_id='us_map', component_property='clickData'),
    Input(component_id='scatter_white_vs_admitted', component_property='relayoutData')

)
def update_parallel_display(pie_clickData,selectedData,parcoord_data,bar_clickData, map_clickData, scatter_zoomData):
    default_color = 'rgba(99, 110, 250, 1)'
    ctx = dash.callback_context

    pcd_df = df

    if ctx.inputs['act_pie_chart.clickData'] is not None:
        # print("pie is not none")
        default_color = pie_clickData['points'][0]['color']
        value_one = int(pie_clickData['points'][0]['customdata'][0].split('-')[0])
        value_two = int(pie_clickData['points'][0]['customdata'][0].split('-')[1])
        pcd_df = pcd_df.loc[(pcd_df['ACT Composite 75th percentile score'] >= value_one) & (pcd_df['ACT Composite 75th percentile score'] <= value_two)]

    if ctx.inputs['scatter_white_vs_admitted.selectedData'] is not None:
        # print("scatter is not None")
        x_value_low = selectedData['range']['x'][0]
        x_value_high = selectedData['range']['x'][1]
        y_value_low = selectedData['range']['y'][0]
        y_value_high = selectedData['range']['y'][1]
        pcd_df = pcd_df.loc[(pcd_df['Percent of total enrollment that are White'] >= x_value_low) &
                            (pcd_df['Percent of total enrollment that are White'] <= x_value_high) &
                            (pcd_df['Percent admitted - total'] >= y_value_low) &
                            (pcd_df['Percent admitted - total'] <= y_value_high)]

    if ctx.inputs['us_map.clickData'] is not None:
        # print("map is not none")
        # print(usmap_clickData['points'][0]['location'])
        location = map_clickData['points'][0]['location']
        pcd_df = pcd_df.loc[(pcd_df['code'] == location)]

    if ctx.inputs['graduation_histogram.clickData'] is not None:
        # print("bar not none")
        bar_value_one = int(bar_clickData['points'][0]['y'].split('-')[0])
        bar_value_two = int(bar_clickData['points'][0]['y'].split('-')[1])
        pcd_df = pcd_df.loc[(pcd_df['Graduation rate - Bachelor degree within 4 years'] >= bar_value_one) & (
            pcd_df['Graduation rate - Bachelor degree within 4 years'] <= bar_value_two)]

    if ctx.inputs['scatter_white_vs_admitted.relayoutData'] is not None:
        # print(scatter_zoomData)
        if 'xaxis.range[0]' in scatter_zoomData:
            x_value_low = scatter_zoomData['xaxis.range[0]']
            x_value_high = scatter_zoomData['xaxis.range[1]']
            y_value_low = scatter_zoomData['yaxis.range[0]']
            y_value_high = scatter_zoomData['yaxis.range[1]']
            pcd_df = pcd_df.loc[(pcd_df['Percent of total enrollment that are White'] >= x_value_low) &
                                (pcd_df['Percent of total enrollment that are White'] <= x_value_high) &
                                (pcd_df['Percent admitted - total'] >= y_value_low) &
                                (pcd_df['Percent admitted - total'] <= y_value_high)]

    if ctx.inputs['parallel_coordinates_display.restyleData'] is not None:
        pcd_df = pcd_dataframe_selecter(pcd_df, parcoord_data)

    fig = go.Figure(data=
        go.Parcoords(
            line_color=default_color,
            dimensions=list([
                dict(range=[300, 800],
                     # constraintrange=[700, 800],  # change this range by dragging the pink line
                     label='SAT Reading', values=list(pcd_df['SAT Critical Reading 75th percentile score'])),
                dict(range=[300, 800],
                     # tickvals=[1.5, 3, 4.5],
                     label='SAT Math', values=list(pcd_df['SAT Critical Reading 75th percentile score'])),
                dict(range=[13, 36],
                     # tickvals=[1, 2, 4, 5],
                     label='ACT Score', values=list(pcd_df['ACT Composite 75th percentile score'])),

                dict(range=[0, 100],
                     label='Graduation Rate - 4 yr', values= list(pcd_df['Graduation rate - Bachelor degree within 4 years'])),
                dict(range=[0, 100],
                     label='% students - black',
                     values=list(pcd_df['Percent of total enrollment that are Black or African American'])),
                dict(range=[0, 100],
                     label='% students - white', values=list(pcd_df['Percent of total enrollment that are White'])),
                dict(range=[0, 100],
                     label='% admitted',
                     values=list(pcd_df['Percent admitted - total'])),
                dict(range=[0, 100],
                     label='% freshman financial aid',
                     values=list(pcd_df['Percent of freshmen receiving any financial aid'])),
                dict(range=[0, 100],
                     label='% women',
                     values=list(pcd_df['Percent of total enrollment that are women'])),
                # dict(range=[0, 65],
                #      label='County Poverty Rate 18 to 34 yo', values=list(df['County Poverty Rate 18 to 34'])),

            ])
        )
    )
    fig.update_layout(
        title_text="Parallel Coordinates Display",
        title_x=0.5
    )

    return fig

@app.callback(
    Output(component_id='scatter_white_vs_admitted', component_property='figure'),
    Input(component_id='parallel_coordinates_display', component_property='restyleData'),
    Input(component_id='graduation_histogram', component_property='clickData'),
    Input(component_id='us_map', component_property='clickData'),
    Input(component_id='scatter_white_vs_admitted',component_property='selectedData'),
    Input(component_id='act_pie_chart', component_property='clickData'),
)
def make_scatter(parcoord_data, bar_clickData, usmap_clickData, scatter_selectedData, pie_clickData):
    ctx = dash.callback_context
    scatter_df = df

    if ctx.inputs['act_pie_chart.clickData'] is not None:
        # print("pie is not none")
        value_one = int(pie_clickData['points'][0]['customdata'][0].split('-')[0])
        value_two = int(pie_clickData['points'][0]['customdata'][0].split('-')[1])
        scatter_df = scatter_df.loc[(scatter_df['ACT Composite 75th percentile score'] >= value_one) & (scatter_df['ACT Composite 75th percentile score'] <= value_two)]

    if ctx.inputs['scatter_white_vs_admitted.selectedData'] is not None:
        # print("scatter is not None")
        x_value_low = scatter_selectedData['range']['x'][0]
        x_value_high = scatter_selectedData['range']['x'][1]
        y_value_low = scatter_selectedData['range']['y'][0]
        y_value_high = scatter_selectedData['range']['y'][1]
        scatter_df = scatter_df.loc[(scatter_df['Percent of total enrollment that are White'] >= x_value_low) &
                            (scatter_df['Percent of total enrollment that are White'] <= x_value_high) &
                            (scatter_df['Percent admitted - total'] >= y_value_low) &
                            (scatter_df['Percent admitted - total'] <= y_value_high)]

    if ctx.inputs['us_map.clickData'] is not None:
        # print("map is not none")
        # print(usmap_clickData['points'][0]['location'])
        location = usmap_clickData['points'][0]['location']
        scatter_df = scatter_df.loc[(scatter_df['code'] == location)]

    if ctx.inputs['graduation_histogram.clickData'] is not None:
        # print("bar not none")
        bar_value_one = int(bar_clickData['points'][0]['y'].split('-')[0])
        bar_value_two = int(bar_clickData['points'][0]['y'].split('-')[1])
        scatter_df = scatter_df.loc[(scatter_df['Graduation rate - Bachelor degree within 4 years'] >= bar_value_one) & (
            scatter_df['Graduation rate - Bachelor degree within 4 years'] <= bar_value_two)]

    if ctx.inputs['parallel_coordinates_display.restyleData'] is not None:
        scatter_df = pcd_dataframe_selecter(scatter_df, parcoord_data)

    fig = px.scatter(scatter_df, x='Percent of total enrollment that are White', y='Percent admitted - total',color='Graduation rate - Bachelor degree within 4 years',
                     labels={'Percent of total enrollment that are White': "Percent of Students that are White (%)",'Percent admitted - total': 'Percent Admittance Rate (%)'},
                     hover_data=['Name','State'])

    fig.update_layout(title="Percent White vs Percent Admitted", title_x=0.5,)
    return fig

@app.callback(
    Output(component_id='act_pie_chart', component_property='figure'),
    Input(component_id='parallel_coordinates_display', component_property='restyleData'),
    Input(component_id='graduation_histogram', component_property='clickData'),
    Input(component_id='us_map', component_property='clickData'),
    Input(component_id='scatter_white_vs_admitted',component_property='selectedData'),
    Input(component_id='act_pie_chart', component_property='clickData'),
    Input(component_id='scatter_white_vs_admitted', component_property='relayoutData')

)
def update_act_piechart(parcoords_data, bar_clickData, usmap_clickData, scatter_selectedData, pie_clickData, scatter_zoomData):
    # print(pie_hoverData)
    pie_df = df
    ctx = dash.callback_context
    # print('pie ctx is')
    # print(ctx.triggered)
    # print(ctx.states)
    # print(ctx.inputs)
    if ctx.inputs['act_pie_chart.clickData'] is not None:
        # print("pie is not none")
        value_one = int(pie_clickData['points'][0]['customdata'][0].split('-')[0])
        value_two = int(pie_clickData['points'][0]['customdata'][0].split('-')[1])
        pie_df = pie_df.loc[(pie_df['ACT Composite 75th percentile score'] >= value_one) & (pie_df['ACT Composite 75th percentile score'] <= value_two)]

    if ctx.inputs['scatter_white_vs_admitted.selectedData'] is not None:
        # print("scatter is not None")
        x_value_low = scatter_selectedData['range']['x'][0]
        x_value_high = scatter_selectedData['range']['x'][1]
        y_value_low = scatter_selectedData['range']['y'][0]
        y_value_high = scatter_selectedData['range']['y'][1]
        pie_df = pie_df.loc[(pie_df['Percent of total enrollment that are White'] >= x_value_low) &
                            (pie_df['Percent of total enrollment that are White'] <= x_value_high) &
                            (pie_df['Percent admitted - total'] >= y_value_low) &
                            (pie_df['Percent admitted - total'] <= y_value_high)]

    if ctx.inputs['us_map.clickData'] is not None:
        # print("map is not none")
        # print(usmap_clickData['points'][0]['location'])
        location = usmap_clickData['points'][0]['location']
        pie_df = pie_df.loc[(pie_df['code'] == location)]
        # print(pie_df)

    if ctx.inputs['graduation_histogram.clickData'] is not None:
        # print("bar not none")
        bar_value_one = int(bar_clickData['points'][0]['y'].split('-')[0])
        bar_value_two = int(bar_clickData['points'][0]['y'].split('-')[1])
        pie_df = pie_df.loc[(pie_df['Graduation rate - Bachelor degree within 4 years'] >= bar_value_one) & (
            pie_df['Graduation rate - Bachelor degree within 4 years'] <= bar_value_two)]

    if ctx.inputs['scatter_white_vs_admitted.relayoutData'] is not None:
        print(scatter_zoomData)
        if 'xaxis.range[0]' in scatter_zoomData:
            x_value_low = scatter_zoomData['xaxis.range[0]']
            x_value_high = scatter_zoomData['xaxis.range[1]']
            y_value_low = scatter_zoomData['yaxis.range[0]']
            y_value_high = scatter_zoomData['yaxis.range[1]']
            pie_df = pie_df.loc[(pie_df['Percent of total enrollment that are White'] >= x_value_low) &
                                (pie_df['Percent of total enrollment that are White'] <= x_value_high) &
                                (pie_df['Percent admitted - total'] >= y_value_low) &
                                (pie_df['Percent admitted - total'] <= y_value_high)]

    if ctx.inputs['parallel_coordinates_display.restyleData'] is not None:
        pie_df = pcd_dataframe_selecter(pie_df, parcoords_data)

    filter_df = pie_df.filter(items=['ACT Composite 75th percentile score']).values.tolist()
    act_categories = ['13-16', '17-20', '21-24', '25-28', '29-32', '33-36']
    distributions = [0] * 6
    for val in filter_df:
        if val[0] <= 16:
            distributions[0] += 1
        elif val[0] <= 20:
            distributions[1] += 1
        elif val[0] <= 24:
            distributions[2] += 1
        elif val[0] <= 28:
            distributions[3] += 1
        elif val[0] <= 32:
            distributions[4] += 1
        else:
            distributions[5] += 1

    # fig = px.histogram(df, x='ACT Composite 75th percentile score')
    act_fig = px.pie(values=distributions, labels=act_categories, color=act_categories,
                     title="Distribution of Universities' ACT Scores 75th percentile",
                     names=act_categories)
    return act_fig

@app.callback(
    Output(component_id='graduation_histogram', component_property='figure'),
    Input(component_id='act_pie_chart', component_property='clickData'),
    Input(component_id='scatter_white_vs_admitted',component_property='selectedData'),
    Input(component_id='parallel_coordinates_display', component_property='restyleData'),
    Input(component_id='us_map', component_property='clickData'),
    Input(component_id='graduation_histogram', component_property='clickData'),
    Input(component_id='scatter_white_vs_admitted', component_property='relayoutData')

)
def update_graduation_histogram(pie_clickData, scatter_selectedData, parcoords_data, usmap_clickData, bar_clickData, scatter_zoomData):

    default_color = 'rgba(99, 110, 250, 1)'
    act_df = df
    ctx = dash.callback_context

    if ctx.inputs['graduation_histogram.clickData'] is not None:
        # print("bar not none")

        bar_value_one = int(bar_clickData['points'][0]['y'].split('-')[0])
        bar_value_two = int(bar_clickData['points'][0]['y'].split('-')[1])
        act_df = act_df.loc[(act_df['Graduation rate - Bachelor degree within 4 years'] >= bar_value_one) & (
            act_df['Graduation rate - Bachelor degree within 4 years'] <= bar_value_two)]

    if ctx.inputs['act_pie_chart.clickData'] is not None:
        # print("pie is not none")
        default_color = pie_clickData['points'][0]['color']
        value_one = int(pie_clickData['points'][0]['customdata'][0].split('-')[0])
        value_two = int(pie_clickData['points'][0]['customdata'][0].split('-')[1])
        act_df = act_df.loc[(act_df['ACT Composite 75th percentile score'] >= value_one) & (act_df['ACT Composite 75th percentile score'] <= value_two)]

    if ctx.inputs['scatter_white_vs_admitted.selectedData'] is not None:
        # print("scatter is not none")
        x_value_low = scatter_selectedData['range']['x'][0]
        x_value_high = scatter_selectedData['range']['x'][1]
        y_value_low = scatter_selectedData['range']['y'][0]
        y_value_high = scatter_selectedData['range']['y'][1]
        act_df = act_df.loc[(act_df['Percent of total enrollment that are White'] >= x_value_low) &
                            (act_df['Percent of total enrollment that are White'] <= x_value_high) &
                            (act_df['Percent admitted - total'] >= y_value_low) &
                            (act_df['Percent admitted - total'] <= y_value_high)]

    # map
    if ctx.inputs['us_map.clickData'] is not None:
        # print("map is not none")
        # print(usmap_clickData['points'][0]['location'])
        location = usmap_clickData['points'][0]['location']
        act_df = act_df.loc[(act_df['code'] == location)]

    if ctx.inputs['scatter_white_vs_admitted.relayoutData'] is not None:
        print(scatter_zoomData)
        if 'xaxis.range[0]' in scatter_zoomData:
            x_value_low = scatter_zoomData['xaxis.range[0]']
            x_value_high = scatter_zoomData['xaxis.range[1]']
            y_value_low = scatter_zoomData['yaxis.range[0]']
            y_value_high = scatter_zoomData['yaxis.range[1]']
            act_df = act_df.loc[(act_df['Percent of total enrollment that are White'] >= x_value_low) &
                                (act_df['Percent of total enrollment that are White'] <= x_value_high) &
                                (act_df['Percent admitted - total'] >= y_value_low) &
                                (act_df['Percent admitted - total'] <= y_value_high)]

    # par coords
    if ctx.inputs['parallel_coordinates_display.restyleData'] is not None:
        act_df = pcd_dataframe_selecter(act_df, parcoords_data)

    filter_df = act_df['Graduation rate - Bachelor degree within 4 years']
    distributions = [0] * 6
    graduation_rates = ['0-16', '17-33', '34-50', '51-67', '68-84', '85-100']
    for val in filter_df:
        if val <= 16:
            distributions[0] += 1
        elif val <= 33:
            distributions[1] += 1
        elif val <= 50:
            distributions[2] += 1
        elif val <= 67:
            distributions[3] += 1
        elif val <= 84:
            distributions[4] += 1
        else:
            distributions[5] += 1

    bar = go.Figure(go.Bar(
        x=distributions,
        y=graduation_rates,
        orientation='h',
        marker_color=default_color))

    bar.update_layout(
        title_text="Graduation Rate - Bachelor Degree in 4 years",
        title_x=0.5,
        xaxis_title="Count",
        yaxis_title="Graduation Rate (%)",


    )
    return bar

@app.callback(
    Output(component_id='us_map', component_property='figure'),
    Input(component_id='act_pie_chart', component_property='clickData'),
    Input(component_id='scatter_white_vs_admitted', component_property='selectedData'),
    Input(component_id='parallel_coordinates_display', component_property='restyleData'),
    Input(component_id='graduation_histogram', component_property='clickData'),
    Input(component_id='us_map', component_property='clickData'),
    Input(component_id='scatter_white_vs_admitted', component_property='relayoutData')
)
def us_map_chart(pie_clickData,scatter_selectedData,parcoords_data, bar_clickData, usmap_clickData, scatter_zoomData):

    ctx = dash.callback_context
    print('map ctx is')
    print(ctx.triggered)
    print(ctx.states)
    print(ctx.inputs)
    # print(ctx.inputs['act_pie_chart.clickData'])
    # print(ctx.inputs['scatter_white_vs_admitted.selectedData'])
    # print(ctx.inputs['parallel_coordinates_display.selectedData'])
    act_df = df

    if ctx.inputs['graduation_histogram.clickData'] is not None:
        # print("bar is not none")
        bar_value_one = int(bar_clickData['points'][0]['y'].split('-')[0])
        bar_value_two = int(bar_clickData['points'][0]['y'].split('-')[1])
        act_df = act_df.loc[(act_df['Graduation rate - Bachelor degree within 4 years'] >= bar_value_one) & (
                    act_df['Graduation rate - Bachelor degree within 4 years'] <= bar_value_two)]

    if ctx.inputs['act_pie_chart.clickData'] is not None:
        # print("pie is not none")
        value_one = int(pie_clickData['points'][0]['customdata'][0].split('-')[0])
        value_two = int(pie_clickData['points'][0]['customdata'][0].split('-')[1])
        act_df = act_df.loc[(act_df['ACT Composite 75th percentile score'] >= value_one) & (
                    act_df['ACT Composite 75th percentile score'] <= value_two)]

    if ctx.inputs['scatter_white_vs_admitted.selectedData'] is not None:
        # print("scatter is not none")
        x_value_low = scatter_selectedData['range']['x'][0]
        x_value_high = scatter_selectedData['range']['x'][1]
        y_value_low = scatter_selectedData['range']['y'][0]
        y_value_high = scatter_selectedData['range']['y'][1]
        act_df = act_df.loc[(act_df['Percent of total enrollment that are White'] >= x_value_low) &
                (act_df['Percent of total enrollment that are White'] <= x_value_high) &
                (act_df['Percent admitted - total'] >= y_value_low) &
                (act_df['Percent admitted - total'] <= y_value_high)]

    if ctx.inputs['us_map.clickData'] is not None:
        location = usmap_clickData['points'][0]['location']
        act_df = act_df.loc[(act_df['code'] == location)]

    if ctx.inputs['scatter_white_vs_admitted.relayoutData'] is not None:
        if 'xaxis.range[0]' in scatter_zoomData:
            x_value_low = scatter_zoomData['xaxis.range[0]']
            x_value_high = scatter_zoomData['xaxis.range[1]']
            y_value_low = scatter_zoomData['yaxis.range[0]']
            y_value_high = scatter_zoomData['yaxis.range[1]']
            act_df = act_df.loc[(act_df['Percent of total enrollment that are White'] >= x_value_low) &
                                (act_df['Percent of total enrollment that are White'] <= x_value_high) &
                                (act_df['Percent admitted - total'] >= y_value_low) &
                                (act_df['Percent admitted - total'] <= y_value_high)]

    if ctx.inputs['parallel_coordinates_display.restyleData'] is not None:
        act_df = pcd_dataframe_selecter(act_df, parcoords_data)

    map_dict = {}
    for index, row in act_df.iterrows():
        # print(row['code'], row['Name'])
        if row['code'] not in map_dict.keys():
            map_dict[row['code']] = 1
        else:
            map_dict[row['code']] += 1
    states = []
    val = []
    for key,value in map_dict.items():
        states.append(key)
        val.append(value)

    # build new df now
    df_dict = {'state': states, 'values': val}
    # map_df = pd.DataFrame(data=df_dict)
    map_df = pd.DataFrame(data=df_dict)

    ########################
    fig = go.Figure(data=go.Choropleth(
        locations=map_df['state'],  # Spatial coordinates
        z=map_df['values'].astype(float),
        # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='Reds',
        colorbar_title="Number Colleges"
    ))

    fig.update_layout(
        title_text='Number of Colleges by State ',
        geo_scope='usa',  # limit map scope to USA
        title_x = 0.5,
    )

    return fig

app.layout = html.Div(

    children=[
        html.Div([
            dcc.Graph(
                id='parallel_coordinates_display',
                # figure=make_parallel_display(),
            )
        ],style={'width':'60%', 'display': 'inline-block'}),

        html.Div(
            [
                dcc.Graph(
                    id='us_map',
                    # figure=us_map_chart()
                )
            ],style={'width':'40%', 'display': 'inline-block'})
        ,
        html.Div(
            [dcc.Graph(
                id='act_pie_chart',
            )],style={'width':'33%', 'display': 'inline-block'}

        ),
        html.Div(
            [dcc.Graph(
                id='graduation_histogram',
                # hoverData=[]
            )],style={'width':'33%', 'display': 'inline-block'}

        ),
        html.Div(
            [
                dcc.Graph(
                    id='scatter_white_vs_admitted',
                )
            ],style={'width':'33%', 'display': 'inline-block'}
        ),

    ]
)

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)