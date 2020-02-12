import requests
import pandas as pd
import numpy as np
#import plotly.graph_objects as go
import math

# Gaussian kernel density estimate for density plot
from scipy.stats import gaussian_kde

# List of lists to single list
from itertools import chain
import random
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


from bokeh.models import (CategoricalColorMapper, HoverTool, ColumnDataSource, Panel, CustomJS,
                          FuncTickFormatter, NumeralTickFormatter, PrintfTickFormatter, SingleIntervalTicker, LinearAxis)
from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider, Tabs, CheckboxButtonGroup, Div,
                                  TableColumn, DataTable, Select, RadioButtonGroup, RadioGroup, DateRangeSlider, Button)

from bokeh.models.widgets import Dropdown

#import param

        
from bokeh.layouts import column, layout, row, WidgetBox, grid
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
def query(endpoint, query, limit = 50000):
    url = 'https://data.cityofnewyork.us/resource/%s'%(endpoint)
    res_df = pd.DataFrame()
    qlimit=10000
    for offset in range(0, limit, qlimit):
        res = requests.get(url, params={'$query': query + " limit %s offset %s" %(qlimit, offset)} )
        if res.status_code != 200:
            raise Exception('{}: for query {}, reason: {}'.format(res.status_code, query, res.content[:200]))
        df = pd.read_csv(BytesIO(res.content), low_memory=False)

        res_df = res_df.append(df, sort=True)
        if len(df.index) < qlimit:
            break
    if res_df.index.size == limit:
        print("warning, query result might be truncated")
    return res_df

approval_res = 'rbx6-tga4.csv'
filings_res = 'ic3t-wcy2.csv'

jobs = ["Plumbing",
"Mechanical",
"Boiler",
"Fuel Burning",
"Fuel Storage",
"Standpipe",
"Sprinkler",
"Fire Alarm",
"Equipment",
"Fire Suppression",
"Curb Cut",
"Other"]
# by construction jobs and jobs col are paired
jobs_cols = [s.lower().replace(' ', '_') for s in jobs]
jobs_to_cols = dict(zip(jobs, jobs_cols))
cols_to_jobs = dict(zip(jobs_cols, jobs))


cols_maps =\
{
    approval_res :
    {
        'date': "issued_date",
        'borough' : "borough",
        'estimated_job_costs': "estimated_job_costs",
        'street_name' : 'street_name',
        'house_no' : 'house_no',
        'work_on_floor' : 'work_on_floor',
        'job_description' : 'job_description',
        'other_description': 'work_type',
    },
    filings_res:
    {
        'date' : 'approved',
        'borough': 'borough',
        'estimated_job_costs' : 'initial_cost',
        'street_name': 'street_name',
        'house_no' : 'house__',
        'work_on_floor' : None,
        'job_description' : 'job_description',
        'other_description': 'other_description',
    }
}

# For Approvals: job_filing_number: Unique id for the filing; first
# character is a letter representing the NYC Borough (B=Brooklyn,
# M=Manhattan, Q=Queens, S=Staten Island, X=Bronx), next 8 characters
# are numbers that make the Job Filing Number unique; last three
# characters are a dash, a letter, and a number. letter is either
# I=Initial, P=Post Approval Amendment, or S=Subsequent. number
# indicates the sequence of the filing; I1 should always exist, you
# can have multiple Subsequent filings, and the number indicates the
# order in which they were filed.

# For filings: job__: This is the unique identifier for the
# application submitted to the Department. It may contain several work
# types, and more work types may be added as the application review
# and the work continues. It is a 9-digit number where the first digit
# indicates the borough where the building is located.  • 1 =
# Manhattan • 2 = Bronx • 3 = Brooklyn • 4 = Queens • 5 = Staten
# Island

mapping={"B":"3", "M":"1", "Q":"4", "S":"5", "X":"2"}   
def remap_ids(id):
    # convert first char to digic and drop last 3 bytes
    return int(mapping[id[0]]+ id[1:9])



assert (cols_maps[approval_res].keys() == cols_maps[filings_res].keys())
cache_cols = list(cols_maps[filings_res].keys()) + jobs_cols
    
def make_empty_date_df():
    df = pd.DataFrame(columns=cache_cols)
    df = df.set_index('date')
    return (set(), df)

query_date_cache=defaultdict(make_empty_date_df)

def save_date_cache(endpoint, fetched_weeks, cache):
    query_date_cache[endpoint] = (fetched_weeks, cache)
    print('saving date cache', '/tmp/date.' + endpoint)
    cache.to_csv('/tmp/date.' + endpoint)
    print('saving date cache done')

def load_date_cache(endpoint):
    fetched_weeks, cache = query_date_cache[endpoint]

    if not fetched_weeks:
        try:
            print('loading date cache', '/tmp/date.' + endpoint)
            new_cache = pd.read_csv('/tmp/date.' + endpoint)
            new_cache['date'] = pd.to_datetime(new_cache['date']).apply(pd.Timestamp)

            new_cache.set_index('date', inplace=True)
            if set(new_cache.columns) != set(cache.columns):
                raise Exception('bad date cache: mismatched columns', sorted(new_cache.columns), sorted(cache.columns))
            cache = new_cache

            fetched_weeks = set(d - relativedelta(weekday=Monday(-1)) for d in cache.index.unique())
            query_date_cache[endpoint] = (fetched_weeks, cache)
            print('loading date cache done (entries: {})'.format(len(cache)))
        except Exception as e:
            print('no cache or bad date cache in ', '/tmp/date.' + endpoint, e)
    return fetched_weeks, cache

def query_approvals(date_start, date_end):
    print(datetime.datetime.now(),'\tquery approvals start')                
    wanted = list(time_range(date_start, date_end))
    
    # we fetch one week at a time
    wanted_weeks = set(d - relativedelta(weekday=Monday(-1)) for d in wanted)
    fetched_weeks, cache = load_date_cache(approval_res)
    old_cache_index = cache.index
    missing_weeks = wanted_weeks - fetched_weeks
    
    # new fetched after fetch
    fetched_weeks = fetched_weeks | missing_weeks

    cols_map = cols_maps[approval_res]
    query_cols = [c for c in cols_map.values() if isinstance(c, str)]
    print(datetime.datetime.now(),'\t\tcomputed missing, count:', len(missing_weeks))
    fetched = 0
    for i in missing_weeks:
        next = i + pd.Timedelta(days=7)
        range = list(time_range(i, next))
        start = i.strftime('%Y-%m-%dT00:00:00.000')
        end = next.strftime('%Y-%m-%dT00:00:00.000')
        q =  "select {cols} where issued_date >= '{start}' "\
                            "and issued_date < '{end}' "\
                            "order by issued_date asc".format(
                                cols=','.join(query_cols),
                                start=start, 
                                end=end)
        print(datetime.datetime.now(),'\t\tget start, range:', i, next )
        source = query(approval_res,q)
        print(datetime.datetime.now(),'\t\tget end, count', len(source))
        result = pd.DataFrame(columns=cache_cols)
        for d, s in cols_map.items():
            result[d] = source[s]

        result['date'] = pd.to_datetime(source['issued_date']).apply(pd.Timestamp)

        for job, job_col in jobs_to_cols.items():
            result[job_col] = False
        result['other'] = ~source['work_type'].isin(jobs)

        result.set_index('date',inplace=True)
        fetched+=len(result)
        cache=cache.append(result, sort=True)
    print(datetime.datetime.now(),'\t\tfetch done')        
    if set(cache.index) != set(old_cache_index):
        cache.sort_index(inplace=True)
        print(datetime.datetime.now(),'\t\tindex sorted')        
        save_date_cache(approval_res, fetched_weeks, cache)
        print(datetime.datetime.now(),'\t\tcache saved')          
    res = cache[date_start:date_end]
    print(datetime.datetime.now(),'\tquery approvals end')
    return res
    #return cache[(cache.index >= date_start) & (cache.index < date_end)]

def query_filings(date_start, date_end):
    print(datetime.datetime.now(),'\tquery filings start')                    
    wanted = list(time_range(date_start, date_end))
    
    # we fetch one week at a time
    wanted_weeks = set(d - relativedelta(weekday=Monday(-1)) for d in wanted)
    fetched_weeks, cache = load_date_cache(filings_res)
    old_cache_index = cache.index
    missing_weeks = wanted_weeks - fetched_weeks
    
    # new fetched after fetch
    fetched_weeks = fetched_weeks | missing_weeks

    cols_map = cols_maps[filings_res]
    cols = [c for c in cols_map.values() if isinstance(c, str)]
    query_cols = cols+jobs_cols
    print(datetime.datetime.now(),'\t\tcomputed missing, count:', len(missing_weeks))
    fetched=0
    for i in missing_weeks:
        next = i + pd.Timedelta(days=7)
        range = list(time_range(i, next))
        start = i.strftime('%m/%d/%Y')
        end= next.strftime('%m/%d/%Y')
        q =  "select {cols} where approved >= '{start}' "\
                            "and approved < '{end}' "\
                            "and building_type != '1-2-3 FAMILY' "\
                            "order by approved asc".format(
                                cols=','.join(query_cols),
                                start=start,
                                end=end)
        print(datetime.datetime.now(),'\t\tget start, range:', i, next )        
        source = query(filings_res, q)
        print(datetime.datetime.now(),'\t\tget end, count', len(source))

        result = pd.DataFrame(columns=cache_cols)

        for d, s in cols_map.items():
            if isinstance(s, str):
                result[d] = source[s]
            else:
                result[d] = None

        result['date'] = pd.to_datetime(source['approved'])\
                           .apply(pd.Timestamp)
        result['estimated_job_costs'] = result['estimated_job_costs']\
            .str.replace('$', '').astype(float)
        for c in jobs_cols:
            result[c] = source[c]=='X'
        result.set_index('date',inplace=True)
        fetched  += len(result)
        cache=cache.append(result, sort=True)
    print(datetime.datetime.now(),'\t\tfetch done')
    if set(cache.index) != set(old_cache_index):
        cache.sort_index(inplace=True)
        print(datetime.datetime.now(),'\t\tindex sorted')
        save_date_cache(filings_res, fetched_weeks, cache)
        print(datetime.datetime.now(),'\t\tcache saved')        
    res = cache[date_start:date_end]
    print(datetime.datetime.now(),'\tquery filings end')                
    return res
    #return cache[(cache.index >= date_start) & (cache.index < date_end)]


# DateRangeSlider has a bug, sometimes value is a pair of Timestamps, sometimes is a float or int (fractional millis since the epoch)
def fixup_date(date):
    if isinstance(date, (int, float)):
            # pandas expects nanoseconds since epoch
            date = pd.Timestamp(float(date)*1e6)
    else:
            date = pd.Timestamp(date)
    return pd.Timestamp(date.date())

# JavaScript code for table download button
download_js=R"""
function table_to_csv(source, df_columns, columns) {
    const nrows = source.get_length()
    const lines = [columns.join(',')]

    for (let i = 0; i < nrows; i++) {
        let row = [];
        for (let j = 0; j < df_columns.length; j++) {
            const column = df_columns[j]
            row.push(source.data[column][i].toString())
        }
        lines.push(row.join(','))
    }
    return lines.join('\n').concat('\n')
}


const filename = 'data_result.csv'
const filetext = table_to_csv(source, df_columns, columns)
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' })

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}
"""    
def modify_doc(doc):
    
    # function to make a dataset for histogram based on a list of set filters

    # stop two days ago, it will take sometime for data to be uploaded
    slider_date_end = datetime.date.today() - relativedelta(hours=48)
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

    last_query=None
    last_query_result=None
    def query_data(date_start, date_end):
        nonlocal last_query
        nonlocal last_query_result
        if last_query == (date_start, date_end):
            return last_query_result
        print(datetime.datetime.now(),'\tstart query', )
        approvals = query_approvals(date_start, date_end)
        
        filings = query_filings(date_start, date_end)
        print(datetime.datetime.now(),'\tend query', )
        assert list(filings.columns) == list(approvals.columns)
        all_df = pd.concat([approvals, filings])
        print(datetime.datetime.now(),'\tend concat', )
        last_query_result = all_df
        last_query = (date_start, date_end)
        return all_df
    
    old_jobs_count = None
    def make_dataset(borough_list, date_start, date_end, bin_width, jobidx):
        print (datetime.datetime.now(), 'start make_dataset')
        nonlocal old_jobs_count
        delta, align = align_range(bin_width)
        date_start += align
        date_end += align + delta
        date_end = min(date_end, pd.Timestamp(datetime.datetime.now().date()))
        all_df = query_data(date_start, date_end)
        if jobidx == 0:
            df = all_df
        else:
            assert old_jobs_count is not None
            old_job_row = old_jobs_count.iloc[jobidx-1]
            if old_job_row['is_categ']:
                # some jobs are categorical
                df = all_df[all_df[old_job_row['job_selector']]]
            else:
                # others we have to look in 'other'
                df = all_df[all_df['other_description'] == old_job_row['job_selector']]
        print(datetime.datetime.now(),'\tjob filtered', )

        def histograms():
            prev_buckets = None
            for i, borough_name in enumerate(borough_list):
                print(datetime.datetime.now(),'\tstart histogram for:', borough_name)
                if borough_name == 'NEW YORK':
                    subset = df
                else:
                    subset = df [df['borough'] == borough_name] 
                print(datetime.datetime.now(),'\t\tborough filtered')
 
                edges = list(time_range(date_start, date_end, delta))
                buckets = subset['estimated_job_costs'].groupby(lambda x: x - align)\
                                                       .agg(sum=np.sum, 
                                                            mean=np.mean,
                                                            median=np.median,
                                                            amax=np.max, 
                                                            len=len)
                print(datetime.datetime.now(),'\t\tbuckets done')
                max_subset = subset.groupby(lambda x: x-align)\
                                   .apply(lambda rows: rows.iloc[np.argmax(rows['estimated_job_costs'].values)])
                print(datetime.datetime.now(),'\t\tmax_subset done')                
                # it is possible that buckets do not cover the full range, so we create 
                # another data frame for the full range and fill it with 0 
                tmp=pd.DataFrame(index=edges, columns=buckets.columns)
                tmp.fillna(0, inplace=True)

                # then we copy the subset shared with the other dataframe
                tmp.loc[buckets.index & tmp.index ] = buckets.loc[buckets.index & tmp.index]
                buckets = tmp
                print(datetime.datetime.now(),'\t\tbuckets fixup done')                            
                # extend edges with an extra 'after-the-end' element
                edges = edges + [edges[-1] + delta]                    
                buckets.sort_index()
                # groupby.agg creates one column per aggregate
                buckets['sum'] /= 10**6
                buckets['mean'] /= 1000
                buckets['median'] /= 1000
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
                    addr = '{borough} {street_name} {house_no} {work_on_floor}'.format(**rows.to_dict())
                    return addr
                if len(max_subset):
                    buckets['f_address'] = max_subset.apply(f_address, axis=1)
                    buckets['f_job_description'] = max_subset['job_description']
                else:
                    buckets['f_address'] = ''
                    buckets['f_job_description'] = ''
                prev_buckets = buckets
                print(datetime.datetime.now(),'\t\tbuckets done')                            
                yield buckets.reset_index()

        # counts of all well knonw jobs (last one is others, don't use it)
        counts = all_df[jobs_cols[:-1]].sum()
        # everything else
        other_counts = all_df['other_description'].value_counts()

        new_job_count = pd.DataFrame({
            'job': [cols_to_jobs[c] for c in counts.index] + [c.title() for c in other_counts.index], 
            'job_selector' : list(counts.index) + list(other_counts.index),
            'is_categ': [True]*len(counts.index) + [False] * len(other_counts.index),
            'count': list(counts.values) + list(other_counts.values)})
        new_job_count['label'] = new_job_count.apply(lambda row: "{job} ({count})".format(**row), axis=1)

        new_job_count = new_job_count.sort_values('count', ascending=False).head(10)                
        # we need to preserve the selected job even if the list of jobs has changed
        if jobidx == 0:
            # easy case: 0 is 'All' and is always on top
            new_job_count = new_job_count.sort_values('count', ascending=False).head(10)
            new_jobidx = 0
        else:
            assert old_jobs_count is not None
            old_job_row = old_jobs_count.iloc[jobidx-1]
            old_job_selector = old_job_row['job_selector']
            new_job_selectors = list(new_job_count['job_selector'])
                    
            if old_job_selector in new_job_selectors:
                # the old job is in the new list
                new_jobidx = new_job_selectors.index(old_job_selector) + 1
            else:
                # artificially add the old job
                row = old_job_row.copy()
                row['label'] = row['job'] # drop the count
                new_jobidx = len(new_job_count)
        assert 0 <= new_jobidx <= len(new_job_count)
                        
        old_jobs_count = new_job_count
        print(datetime.datetime.now(),'\tend counts', )        
        #Dataframe to hold information
        by_borough = pd.DataFrame()
        # Overall dataframe
        all_buckets = list(histograms())
        print(datetime.datetime.now(),'\tend histograms', )        
        by_borough = by_borough.append(all_buckets, sort=False)
        print(datetime.datetime.now(),'\tend append', )        
        by_borough.sort_values(['name', 'left'], inplace=True)
        print(datetime.datetime.now(),'\tend sort', )        
        return ColumnDataSource(by_borough), ['All Jobs'] + list(new_job_count['label']), new_jobidx

    def make_plot(src, title, y_label, tooltip, column):
        # Blank plot with correct labels
        p = figure(title = title,
                   x_axis_type='datetime',
                   x_axis_label = 'Date', y_axis_label = y_label)            
        # Quad glyphs to create a histogram
        p.quad(source = src, bottom = column +'_bottom', top = column + '_top', left = 'left', right = 'right',
               color = 'color', fill_alpha = 0.7, hover_fill_color = 'color', legend_group = 'name',
               hover_fill_alpha = 1.0, line_color = 'black')
        p.legend.background_fill_alpha=0.5
        p.legend.location='bottom_left'
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
    def do_update(force=False):
        try:
            new_params = ([borough_selection.labels[i] for i in borough_selection.active],
                          fixup_date(date_select.value[0]),
                          fixup_date(date_select.value[1]),
                          binwidth_select.value,
                          jobs_selection.menu.index(jobs_selection.value))
            if new_params != old_params[0]:
                show_spinner()
                new_data, jobs_selection_labels, jobs_selection_active = make_dataset(*new_params)
                jobs_selection.menu = jobs_selection_labels
                jobs_selection.value = jobs_selection_labels[jobs_selection_active]
                old_params[0] = new_params
                src.data.update(new_data.data)
        except Exception:
            print(traceback.print_exc())

    def update(attr, old, new):
        print('update called')
        do_update()
    
    # DateRangeSlider mouseup is broken, do nothing on change and use a timer
    slow_update=[time.time()]
    def update_no_op(attr, old, new):
        show_spinner()
        if time.time()-slow_update[0] < .5:
            return
        slow_update[0] = time.time()
        #update(attr, old, new)
    def time_update():
        #return
        slow_update[0] = time.time()
        do_update()
        hide_spinner()
    
    spinner_text = """
<!-- https://www.w3schools.com/howto/howto_css_loader.asp -->
<div class="loader">
<style scoped>
.loader {
    border: 16px solid #f3f3f3; /* Light grey */
    border-top: 16px solid #3498db; /* Blue */
    border-radius: 50%;
    width: 120px;
    height: 120px;
    margin: auto;
    animation: spin 2s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
} 
</style>
</div>
"""

    div_spinner = Div(text=spinner_text, width=120, height=120)
    div_spinner.sizing_mode = "stretch_both"

    binwidth_select = Dropdown(label='week', menu=['day', 'week', 'month']) #index of 'week', i.e. 0
    binwidth_select.value = 'week'
    def on_click(attr, old, new):
        binwidth_select.label = binwidth_select.value
        update(attr, old, new)
    binwidth_select.on_change('value', on_click)
    
    date_default_end= slider_date_end
    date_default_start = date_default_end - relativedelta(months=1)
    date_select = DateRangeSlider(start=slider_date_start, 
                                  end=slider_date_end, 
                                  value=(date_default_start,date_default_end), 
                                  callback_policy='mouseup', # do not start untill mouse released
                                  step=1,
                                  callback_throttle=1000)
    date_select.on_change('value', update_no_op) # this just enables the spinner
    # workaround broken callback_policy for DateRangeSlider
    borough_selection_labels = ['NEW YORK','QUEENS', 'MANHATTAN', 'STATEN ISLAND', 'BROOKLYN', 'BRONX']
    borough_selection = CheckboxGroup(labels=borough_selection_labels,
                                      active = list(range(1, len(borough_selection_labels))),
                                      inline=True)
    borough_selection.on_change('active', update)

    jobs_selection_labels = ['All Jobs']
    # jobs_selection = RadioGroup(labels=jobs_selection_labels, active =  0, orientation='vertical')
    jobs_selection = Dropdown(label = 'All Jobs', menu = ['All Jobs'])
    jobs_selection.value = 'All Jobs'
    def on_click(attr, old, new):
        jobs_selection.label = jobs_selection.value
        update(attr, old, new)
    jobs_selection.on_change('value', on_click)

    #jobs_selection.sizing_mode='stretch_both'
    col_meta = { 
        'len': '%d', 
        'mean': '%dk',
        'median': '%dk',
        'sum': '%dM',
        'amax': '%dk'
    }
    data = [ ('Number of Projects', 'Total projects', 'counts', 'len'),
             ('Most Expensive Project', 'Max cost', 'cost', 'amax'),
             ('Total Project Cost', 'Total project cost', 'cost', 'sum'),
             ('Mean Project Cost', 'Mean project cost', 'cost', 'mean'),
             ('Median Project Cost', 'Median project cost', 'cost', 'median'),]

    tabs_ref = None
    def show_spinner():
        if tabs_ref:
            tabs_ref.tabs = tabs_spinner
    def hide_spinner():
        if tabs_ref:
            tabs_ref.tabs = tabs_content

    do_update()
    plots = [ make_plot(src, *args) for args in data ]

    for p in plots:
        p.sizing_mode='stretch_both'

    table_column_names = [
        ('f_period', 'Period'),
        ('name', 'Borough'),
        ('f_len', 'Projects #'),
        ('f_sum', 'Total Cost'),
        ('f_mean', 'Mean Cost'),
        ('f_median','Median Cost'),
        ('f_amax','Max Cost'),
        ('f_address', 'Max Cost Project Address'),
        ('f_job_description', 'Max Cost Project Description'),
    ]
                
    download_button = Button(label="Download", button_type="success")
    download_button.js_on_click(CustomJS(args=dict(source=src,
                                                   df_columns = [ x for x, _ in table_column_names],
                                                   columns = [ y for _, y in table_column_names]),
                                code=download_js))
    legend_button = Button(label="Hide Legend")
    def toggle_legend(*args):
        if legend_button.label == "Hide Legend":
            legend_button.label = "Show Legend"
        else:
            legend_button.label = "Hide Legend"
        for p in plots:
            p.legend.visible = not p.legend.visible
    legend_button.on_click(toggle_legend)
    #row(column(borough_selection,row(binwidth_select, date_select)), div_spinner, sizing_mode='stretch_both'),
    instrument_row = row(borough_selection,
                         column(row(binwidth_select, jobs_selection, download_button, legend_button),
                                date_select, sizing_mode='stretch_width'), sizing_mode='stretch_width')
    plot_grid = grid(plots +[None], ncols=3)
    plot_tab = Panel(child=plot_grid, title = 'Histogram')
    
    table_columns = [TableColumn(field=f, title=t) for f, t in table_column_names]
    data_table = DataTable(columns=table_columns, source=src)
    data_table.sizing_mode='stretch_both'
    data_tab = Panel(child=data_table, title='Data')

    
    
    tabs_content=[plot_tab, data_tab]
    spinner_grid=grid([div_spinner], ncols=1,
                      sizing_mode='stretch_both')
    tabs_spinner = [Panel(child=spinner_grid, title='Histogram'), Panel(child=spinner_grid, title='Data')]
    
    tabs = Tabs(tabs=tabs_spinner)
    tabs.sizing_mode='stretch_both'
    tabs_ref = tabs
    col = column(
        instrument_row,
        #plot_grid,
        tabs,
        sizing_mode='stretch_both')
    root = col
    doc.add_root(root)

    doc.add_periodic_callback(time_update, 1000)


modify_doc(curdoc())
