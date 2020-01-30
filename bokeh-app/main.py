
import requests
import pandas as pd
import numpy as np
#import plotly.graph_objects as go
import math

# Gaussian kernel density estimate for density plot
from scipy.stats import gaussian_kde

# List of lists to single list
from itertools import chain

import traceback
import requests
import datetime
import time
from io import BytesIO
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from dateutil.relativedelta import MO as Monday 

from bisect import bisect_left
# Bokeh plotting tools
from bokeh.io import show, output_notebook, push_notebook, curdoc
from bokeh.plotting import figure


from bokeh.models import (CategoricalColorMapper, HoverTool, ColumnDataSource, Panel, 
                          FuncTickFormatter, NumeralTickFormatter, PrintfTickFormatter, SingleIntervalTicker, LinearAxis)
from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider, Tabs, CheckboxButtonGroup, Div,
                                  TableColumn, DataTable, Select, RadioButtonGroup, DateRangeSlider)

#import param

        
from bokeh.layouts import column, layout, row, WidgetBox
from bokeh.palettes import Category20_16

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application

# enumerates the half open range [b, b+delta, b+2*delta..., e)
# similar to standard range, which does not work with dates
def time_range(b, e, delta=pd.Timedelta(days=1)):
    i = b
    while i < e:
        yield i
        n = i + delta
        if n == i:
            raise Exception("No progress") # avoid infinite loops
        i = n

# query cityofnewyork specific OData endpoint using 'query' as $query string
# note: will fetch at most 'limit' rows (possibly using paging)
def query(endpoint, query, limit = 10000):
    url = 'https://data.cityofnewyork.us/resource/%s'%(endpoint)
    res_df = pd.DataFrame()
    qlimit=1000
    for offset in range(0, limit, qlimit):

        res = requests.get(url, params={'$query': query + " limit %s offset %s" %(qlimit, offset)} )
        if res.status_code != 200:
            raise Exception('{}: for query {}, reason: {}'.format(res.status_code, query, res.content[:200]))
        df = pd.read_csv(BytesIO(res.content))

        res_df = res_df.append(df, sort=True)
        if len(df.index) < qlimit:
            break
    if res_df.index.size == limit:
        print("warning, query result might be truncated")
    return res_df

approval_res = 'rbx6-tga4.csv'
filings_res = 'ic3t-wcy2.csv'

query_cols = ["issued_date", "borough", "estimated_job_costs"] +\
             ['street_name', 'house_no', 'work_on_floor', 'job_description']

query_cache=defaultdict(lambda : (set(), pd.DataFrame(columns=query_cols).set_index('issued_date')))
def query_dates(endpoint, date_start, date_end):
    fetched = 0
    
    wanted = list(time_range(date_start, date_end))
    
    # we fetch one week at a time
    wanted_weeks = set(d - relativedelta(weekday=Monday(-1)) for d in wanted)
    fetched_weeks, cache = query_cache[endpoint]
    missing_weeks = wanted_weeks - fetched_weeks
    
    # new fetched after fetch
    fetched_weeks = fetched_weeks | missing_weeks
    
    for i in missing_weeks:
        next = i + pd.Timedelta(days=7)
        start = i.strftime('%Y-%m-%dT00:00:00.000')
        end = next.strftime('%Y-%m-%dT00:00:00.000')
        range = list(time_range(i, next))
        df = query(endpoint, "select {cols} where issued_date >= '{start}' "\
                            "and issued_date < '{end}' "
                            "order by issued_date asc".format(
                                cols=','.join(query_cols),
                                start=start, 
                                end=end))
        print('Requested {} - {}; Received rows: {}'.format(start, end, len(df)))
        df['issued_date'] = pd.to_datetime(df['issued_date']).apply(pd.Timestamp)
        df.set_index('issued_date',inplace=True)
        cache=cache.append(df, sort=True)
    cache.sort_index(inplace=True)
    query_cache[endpoint] = (fetched_weeks, cache)
    return cache[(cache.index >= date_start) & (cache.index < date_end)]

# DateRangeSlider has a bug, sometimes value is a pair of Timestamps, sometimes is a float or int (fractional millis since the epoch)
def fixup_date(date):
    if isinstance(date, (int, float)):
            # pandas expects nanoseconds since epoch
            date = pd.Timestamp(float(date)*1e6)
    else:
            date = pd.Timestamp(date)
    return pd.Timestamp(date.date())

    
def modify_doc(doc):
    
    # function to make a dataset for histogram based on a list of set filters

    valid_bin_widths = ['day', 'week', 'month']
    default_bin_width='week'
    slider_date_end = datetime.date.today()
    slider_date_start = slider_date_end - relativedelta(months=6, day=1) # at most 2 months ago    
    
    # return delta and align for a range according to bin_width
    # bin_width is one of 'week', 'month', 'day'
    # delta can be used to move a date to the next bin, align to
    # snap back a range the the current bin start 
    def align_range(bin_width):        
        if bin_width == 'week':
            delta = relativedelta(weeks=1)
            align = relativedelta(weekday=Monday(-1))

        elif bin_width == 'month':
            delta = relativedelta(months=1)
            align = relativedelta(day=1)
        else:
            #nothing special to do for 'day'
            delta = relativedelta(days=1)
            align = relativedelta()

        return delta, align


    def make_dataset(endpoint, borough_list, date_start, date_end, bin_width):
        delta, align = align_range(bin_width)
        date_start += align
        date_end += align + delta
        df = query_dates(endpoint, date_start, date_end)

        def histograms():
            prev_buckets = None
            for i, borough_name in enumerate(borough_list): 
                subset = df [df['borough'] == borough_name]
 
                edges = list(time_range(date_start, date_end, delta))
                buckets = subset['estimated_job_costs'].groupby(lambda x: x - align)\
                                                       .agg(sum=np.sum, 
                                                            mean=np.mean, 
                                                            amax=np.max, 
                                                            len=len)

                max_subset = subset.groupby(lambda x: x-align)\
                                   .apply(lambda rows: rows.iloc[np.argmax(rows['estimated_job_costs'].values)])
                
                # it is possible that buckets do not cover the full range, so we create 
                # another data frame for the full range and fill it with 0 
                tmp=pd.DataFrame(index=edges, columns=buckets.columns)
                tmp.fillna(0, inplace=True)

                # then we copy the subset shared with the other dataframe
                tmp.loc[buckets.index & tmp.index ] = buckets.loc[buckets.index & tmp.index]
                buckets = tmp
            
                # extend edges with an extra 'after-the-end' element
                edges = edges + [edges[-1] + delta]                    
                buckets.sort_index()
                # groupby.agg creates one column per aggregate
                buckets['sum'] /= 10**6
                buckets['mean'] /= 1000
                buckets['amax'] /= 1000
                # nothing to do with buckets['len']
                buckets['left'] = edges[:-1]
                buckets['right'] = edges[1:]
                buckets['color'] = Category20_16[i]
                buckets['name'] = borough_name

                for c, format in col_meta.items():
                    if prev_buckets is not None:
                        buckets[c + '_top'] =  buckets[c] + prev_buckets[c + '_top']
                        buckets[c + '_bottom'] =  prev_buckets[c + '_top']
                    else:
                        buckets[c + '_top'] = buckets[c]
                        buckets[c + '_bottom'] = 0
                    buckets['f_' + c] = buckets[c].apply(lambda x: format%(x))
                buckets['f_period'] = buckets.index.map(lambda x: '{} - {}'.format(x.date(), (x+delta).date()))
                def f_address(rows):
                    addr = '{street_name} {house_no} {work_on_floor}'.format(**rows.to_dict())
                    return addr
                buckets['f_address'] = max_subset.apply(f_address, axis=1)
                buckets['f_job_description'] = max_subset['job_description']
                prev_buckets = buckets

                yield buckets.reset_index()

        #Dataframe to hold information
        by_borough = pd.DataFrame()
        # Overall dataframe
        all_buckets = list(histograms())
        by_borough = by_borough.append(all_buckets, sort=False)
        by_borough.sort_values(['name', 'left'], inplace=True)
        return ColumnDataSource(by_borough)

    def make_plot(src, title, y_label, tooltip, column):
        # Blank plot with correct labels
        p = figure(plot_width = 500, plot_height = 500, 
                   title = title,
                   x_axis_type='datetime',
                   sizing_mode='stretch_both',
                   x_axis_label = 'Date', y_axis_label = y_label)            
        # Quad glyphs to create a histogram
        p.quad(source = src, bottom = column +'_bottom', top = column + '_top', left = 'left', right = 'right',
               color = 'color', fill_alpha = 0.7, hover_fill_color = 'color', legend_label = 'name',
               hover_fill_alpha = 1.0, line_color = 'black')
        
                          
        if column == 'amax':
            tooltips = [('Period:','@f_period'),
                        ('Borough', '@name'), 
                        ('Address', '@f_address'),
                        ('Description', '@f_job_description'),
                        ('cost', '@f_amax')
                    ]
        else:
            tooltips = [('Period:','@f_period'),
                        ('Borough', '@name'), 
                        (tooltip, '@f_'+column)
                    ]
        
        # Hover tool with vline mode
        hover = HoverTool(tooltips=tooltips)

        p.add_tools(hover)

        # Styling
        p = style(p, col_meta[column])

        return p

    def style(p, y_format):
        # Title 
        p.title.align = 'center'
        p.title.text_font_size = '20pt'
        p.title.text_font = 'serif'

        # Axis titles
        p.xaxis.axis_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_style = 'bold'

        p.yaxis.formatter = PrintfTickFormatter (format=y_format)

        # Tick labels
        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'

        return p

    
    src = ColumnDataSource()
    old_params = [None]
    def do_update():
        try:
            new_params = (approval_res, 
                          [borough_selection.labels[i] for i in borough_selection.active],
                          fixup_date(date_select.value[0]),
                          fixup_date(date_select.value[1]),
                          valid_bin_widths[binwidth_select.active])
            if new_params != old_params[0]:
                show_spinner()
                new_data = make_dataset(*new_params)
                old_params[0] = new_params

                src.data.update(new_data.data)
        except Exception:
            print(traceback.print_exc())

    def update(attr, old, new):
        do_update()
    
    # DateRangeSlider mouseup is broken, do nothing on change and use a timer
    slow_update=[time.time()]
    def update_no_op(attr, old, new):
        show_spinner()
        if time.time()-slow_update[0] < .5:
            return
        slow_update[0] = time.time()
        update(attr, old, new)
    def time_update():
        #return
        slow_update[0] = time.time()
        do_update()
        hide_spinner()
    
    spinner_text = """
    <!-- https://www.w3schools.com/howto/howto_css_loader.asp -->
    <div class="loader" >
    <style scoped>
    .loader {
        border: 16px solid #f3f3f3; /* Light grey */
        border-top: 16px solid #3498db; /* Blue */
        border-radius: 50%;
        margin: auto;
        width: 100px;
        height: 100px;
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    } 
    </style>
    </div>
    """
    div_spinner = Div(text="",width=120,height=120)
    def show_spinner():
        div_spinner.text = spinner_text
    def hide_spinner():
        div_spinner.text = ""

    binwidth_select = RadioButtonGroup(labels=valid_bin_widths,
                                       active=valid_bin_widths.index(default_bin_width), #index of 'week', i.e. 0
                                       sizing_mode='stretch_both')
    binwidth_select.on_change('active', update)
    
    date_default_end= slider_date_end
    date_default_start = date_default_end - relativedelta(months=1)
    date_select = DateRangeSlider(start=slider_date_start, 
                                  end=slider_date_end, 
                                  value=(date_default_start,date_default_end), 
                                  callback_policy='mouseup', # do not start untill mouse released
                                  step=1,
                                  callback_throttle=1000,
                                  sizing_mode='stretch_both') # this is slow, so calls at most every 2000ms

    date_select.on_change('value', update_no_op)

    available_boroughs = ['QUEENS', 'MANHATTAN', 'STATEN ISLAND', 'BROOKLYN', 'BRONX']

    borough_selection = CheckboxGroup(labels=available_boroughs, active = list(range(0, len(available_boroughs))),
                                     sizing_mode='stretch_both')
    borough_selection.on_change('active', update)
    
    initial_borough = [borough_selection.labels[i] for i in borough_selection.active]
    
    # Put controls in a single element
    controls = layout([[borough_selection, binwidth_select, date_select, div_spinner]] , width=500)
    
    col_meta = { 
        'len': '%d', 
        'mean': '%dl',
        'sum': '%dM',
        'amax': '%dk'
    }
    
    data = [ ('Number of Projects', 'Total projects', 'counts', 'len'),
             ('Most Expensive Project', 'Max cost', 'cost', 'amax'),
             ('Total Project Cost', 'Total project cost', 'cost', 'sum'),
             ('Mean Project Cost', 'Median project cost', 'cost', 'mean') ]
    do_update()
    plots = [ make_plot(src, *args) for args in data ]

    # Create a row layout
    lyt = layout([controls, plots[3]], 
                 plots[0:3])
    
    # Make a tab with the layout 
    tab = Panel(child=lyt, title = 'Histogram')
    tabs = Tabs(tabs=[tab])
    
    doc.add_periodic_callback(time_update, 1000)
    doc.add_root(tabs)

# Set up an application
#handler = FunctionHandler(modify_doc)
#app = Application(handler)
    
# run the app
#show(app)

modify_doc(curdoc())
