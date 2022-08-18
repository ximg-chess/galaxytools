#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:36:22 2021

@author: rv43
"""

import logging

import os
import sys
import re
import yaml
try:
    import h5py
except:
    pass
import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
except:
    pass

from ast import literal_eval
from copy import deepcopy
from time import time


def depth_list(L): return isinstance(L, list) and max(map(depth_list, L))+1
def depth_tuple(T): return isinstance(T, tuple) and max(map(depth_tuple, T))+1
def unwrap_tuple(T):
    if depth_tuple(T) > 1 and len(T) == 1:
        T = unwrap_tuple(*T)
    return T
   
def illegal_value(value, name, location=None, exit_flag=False):
    if not isinstance(location, str):
        location = ''
    else:
        location = f'in {location} '
    if isinstance(name, str):
        logging.error(f'Illegal value for {name} {location}({value}, {type(value)})')
    else:
        logging.error(f'Illegal value {location}({value}, {type(value)})')
    if exit_flag:
        raise ValueError

def is_int(v, v_min=None, v_max=None):
    """Value is an integer in range v_min <= v <= v_max.
    """
    if not isinstance(v, int):
        return False
    if v_min is not None and not isinstance(v_min, int):
        illegal_value(v_min, 'v_min', 'is_int') 
        return False
    if v_max is not None and not isinstance(v_max, int):
        illegal_value(v_max, 'v_max', 'is_int') 
        return False
    if v_min is not None and v_max is not None and v_min > v_max:
        logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
        return False
    if (v_min is not None and v < v_min) or (v_max is not None and v > v_max):
        return False
    return True

def is_int_pair(v, v_min=None, v_max=None):
    """Value is an integer pair, each in range v_min <= v[i] <= v_max or 
           v_min[i] <= v[i] <= v_max[i].
    """
    if not (isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], int) and
            isinstance(v[1], int)):
        return False
    if v_min is not None or v_max is not None:
        if (v_min is None or isinstance(v_min, int)) and (v_max is None or isinstance(v_max, int)):
            if True in [True if not is_int(vi, v_min=v_min, v_max=v_max) else False for vi in v]:
                return False
        elif is_int_pair(v_min) and is_int_pair(v_max):
            if True in [True if v_min[i] > v_max[i] else False for i in range(2)]:
                logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
                return False
            if True in [True if not is_int(v[i], v_min[i], v_max[i]) else False for i in range(2)]:
                return False
        elif is_int_pair(v_min) and (v_max is None or isinstance(v_max, int)):
            if True in [True if not is_int(v[i], v_min=v_min[i], v_max=v_max) else False
                    for i in range(2)]:
                return False
        elif (v_min is None or isinstance(v_min, int)) and is_int_pair(v_max):
            if True in [True if not is_int(v[i], v_min=v_min, v_max=v_max[i]) else False
                    for i in range(2)]:
                return False
        else:
            logging.error(f'Illegal v_min or v_max input ({v_min} {type(v_min)} and '+
                    f'{v_max} {type(v_max)})')
            return False
    return True

def is_int_series(l, v_min=None, v_max=None):
    """Value is a tuple or list of integers, each in range v_min <= l[i] <= v_max.
    """
    if v_min is not None and not isinstance(v_min, int):
        illegal_value(v_min, 'v_min', 'is_int_series') 
        return False
    if v_max is not None and not isinstance(v_max, int):
        illegal_value(v_max, 'v_max', 'is_int_series') 
        return False
    if not isinstance(l, (tuple, list)):
        return False
    if True in [True if not is_int(v, v_min=v_min, v_max=v_max) else False for v in l]:
        return False
    return True

def is_num(v, v_min=None, v_max=None):
    """Value is a number in range v_min <= v <= v_max.
    """
    if not isinstance(v, (int, float)):
        return False
    if v_min is not None and not isinstance(v_min, (int, float)):
        illegal_value(v_min, 'v_min', 'is_num') 
        return False
    if v_max is not None and not isinstance(v_max, (int, float)):
        illegal_value(v_max, 'v_max', 'is_num') 
        return False
    if v_min is not None and v_max is not None and v_min > v_max:
        logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
        return False
    if (v_min is not None and v < v_min) or (v_max is not None and v > v_max):
        return False
    return True

def is_num_pair(v, v_min=None, v_max=None):
    """Value is a number pair, each in range v_min <= v[i] <= v_max or 
           v_min[i] <= v[i] <= v_max[i].
    """
    if not (isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], (int, float)) and
            isinstance(v[1], (int, float))):
        return False
    if v_min is not None or v_max is not None:
        if ((v_min is None or isinstance(v_min, (int, float))) and
                (v_max is None or isinstance(v_max, (int, float)))):
            if True in [True if not is_num(vi, v_min=v_min, v_max=v_max) else False for vi in v]:
                return False
        elif is_num_pair(v_min) and is_num_pair(v_max):
            if True in [True if v_min[i] > v_max[i] else False for i in range(2)]:
                logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
                return False
            if True in [True if not is_num(v[i], v_min[i], v_max[i]) else False for i in range(2)]:
                return False
        elif is_num_pair(v_min) and (v_max is None or isinstance(v_max, (int, float))):
            if True in [True if not is_num(v[i], v_min=v_min[i], v_max=v_max) else False
                    for i in range(2)]:
                return False
        elif (v_min is None or isinstance(v_min, (int, float))) and is_num_pair(v_max):
            if True in [True if not is_num(v[i], v_min=v_min, v_max=v_max[i]) else False
                    for i in range(2)]:
                return False
        else:
            logging.error(f'Illegal v_min or v_max input ({v_min} {type(v_min)} and '+
                    f'{v_max} {type(v_max)})')
            return False
    return True

def is_num_series(l, v_min=None, v_max=None):
    """Value is a tuple or list of numbers, each in range v_min <= l[i] <= v_max.
    """
    if v_min is not None and not isinstance(v_min, (int, float)):
        illegal_value(v_min, 'v_min', 'is_num_series') 
        return False
    if v_max is not None and not isinstance(v_max, (int, float)):
        illegal_value(v_max, 'v_max', 'is_num_series') 
        return False
    if not isinstance(l, (tuple, list)):
        return False
    if True in [True if not is_num(v, v_min=v_min, v_max=v_max) else False for v in l]:
        return False
    return True

def is_index(v, v_min=0, v_max=None):
    """Value is an array index in range v_min <= v < v_max.
       NOTE v_max IS NOT included!
    """
    if isinstance(v_max, int):
        if v_max <= v_min:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return False
        v_max -= 1
    return is_int(v, v_min, v_max)

def is_index_range(v, v_min=0, v_max=None):
    """Value is an array index range in range v_min <= v[0] <= v[1] <= v_max.
       NOTE v_max IS included!
    """
    if not is_int_pair(v):
        return False
    if not isinstance(v_min, int):
        illegal_value(v_min, 'v_min', 'is_index_range') 
        return False
    if v_max is not None:
        if not isinstance(v_max, int):
            illegal_value(v_max, 'v_max', 'is_index_range') 
            return False
        if v_max < v_min:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return False
    if not v_min <= v[0] <= v[1] or (v_max is not None and v[1] > v_max):
        return False
    return True

def index_nearest(a, value):
    a = np.asarray(a)
    if a.ndim > 1:
        logging.warning(f'Illegal input array ({a}, {type(a)})')
    # Round up for .5
    value *= 1.0+sys.float_info.epsilon
    return (int)(np.argmin(np.abs(a-value)))

def index_nearest_low(a, value):
    a = np.asarray(a)
    if a.ndim > 1:
        logging.warning(f'Illegal input array ({a}, {type(a)})')
    index = int(np.argmin(np.abs(a-value)))
    if value < a[index] and index > 0:
        index -= 1
    return index

def index_nearest_upp(a, value):
    a = np.asarray(a)
    if a.ndim > 1:
        logging.warning(f'Illegal input array ({a}, {type(a)})')
    index = int(np.argmin(np.abs(a-value)))
    if value > a[index] and index < a.size-1:
        index += 1
    return index

def round_to_n(x, n=1):
    if x == 0.0:
        return 0
    else:
        return round(x, n-1-int(np.floor(np.log10(abs(x)))))

def round_up_to_n(x, n=1):
    xr = round_to_n(x, n)
    if abs(x/xr) > 1.0:
        xr += np.sign(x)*10**(np.floor(np.log10(abs(x)))+1-n)
    return xr

def trunc_to_n(x, n=1):
    xr = round_to_n(x, n)
    if abs(xr/x) > 1.0:
        xr -= np.sign(x)*10**(np.floor(np.log10(abs(x)))+1-n)
    return xr

def string_to_list(s):
    """Return a list of numbers by splitting/expanding a string on any combination of
       dashes, commas, and/or whitespaces
       e.g: '1, 3, 5-8,12 ' -> [1, 3, 5, 6, 7, 8, 12]
    """
    if not isinstance(s, str):
        illegal_value(s, location='string_to_list') 
        return None
    if not len(s):
        return []
    try:
        list1 = [x for x in re.split('\s+,\s+|\s+,|,\s+|\s+|,', s.strip())]
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    try:
        l = []
        for l1 in list1:
            l2 = [literal_eval(x) for x in re.split('\s+-\s+|\s+-|-\s+|\s+|-', l1)]
            if len(l2) == 1:
                l += l2
            elif len(l2) == 2 and l2[1] > l2[0]:
                l += [i for i in range(l2[0], l2[1]+1)]
            else:
                raise ValueError
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return sorted(set(l))

def get_trailing_int(string):
    indexRegex = re.compile(r'\d+$')
    mo = indexRegex.search(string)
    if mo is None:
        return None
    else:
        return int(mo.group())

def input_int(s=None, v_min=None, v_max=None, default=None, inset=None):
    if default is not None:
        if not isinstance(default, int):
            illegal_value(default, 'default', 'input_int') 
            return None
        default_string = f' [{default}]'
    else:
        default_string = ''
    if v_min is not None:
        if not isinstance(v_min, int):
            illegal_value(v_min, 'v_min', 'input_int') 
            return None
        if default is not None and default < v_min:
            logging.error('Illegal v_min, default combination ({v_min}, {default})')
            return None
    if v_max is not None:
        if not isinstance(v_max, int):
            illegal_value(v_max, 'v_max', 'input_int') 
            return None
        if v_min is not None and v_min > v_max:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return None
        if default is not None and default > v_max:
            logging.error('Illegal default, v_max combination ({default}, {v_max})')
            return None
    if inset is not None:
        if (not isinstance(inset, (tuple, list)) or False in [True if isinstance(i, int) else
                False for i in inset]):
            illegal_value(inset, 'inset', 'input_int') 
            return None
    if v_min is not None and v_max is not None:
        v_range = f' ({v_min}, {v_max})'
    elif v_min is not None:
        v_range = f' (>= {v_min})'
    elif v_max is not None:
        v_range = f' (<= {v_max})'
    else:
        v_range = ''
    if s is None:
        print(f'Enter an integer{v_range}{default_string}: ')
    else:
        print(f'{s}{v_range}{default_string}: ')
    try:
        i = input()
        if isinstance(i, str) and not len(i):
            v = default
            print(f'{v}')
        else:
            v = literal_eval(i)
        if inset and v not in inset:
           raise ValueError(f'{v} not part of the set {inset}')
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        v = None
    except:
        print('Unexpected error')
        raise
    if not is_int(v, v_min, v_max):
        print('Illegal input, enter a valid integer')
        v = input_int(s, v_min, v_max, default)
    return v

def input_num(s=None, v_min=None, v_max=None, default=None):
    if default is not None:
        if not isinstance(default, (int, float)):
            illegal_value(default, 'default', 'input_num') 
            return None
        default_string = f' [{default}]'
    else:
        default_string = ''
    if v_min is not None:
        if not isinstance(v_min, (int, float)):
            illegal_value(vmin, 'vmin', 'input_num') 
            return None
        if default is not None and default < v_min:
            logging.error('Illegal v_min, default combination ({v_min}, {default})')
            return None
    if v_max is not None:
        if not isinstance(v_max, (int, float)):
            illegal_value(vmax, 'vmax', 'input_num') 
            return None
        if v_min is not None and v_max < v_min:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return None
        if default is not None and default > v_max:
            logging.error('Illegal default, v_max combination ({default}, {v_max})')
            return None
    if v_min is not None and v_max is not None:
        v_range = f' ({v_min}, {v_max})'
    elif v_min is not None:
        v_range = f' (>= {v_min})'
    elif v_max is not None:
        v_range = f' (<= {v_max})'
    else:
        v_range = ''
    if s is None:
        print(f'Enter a number{v_range}{default_string}: ')
    else:
        print(f'{s}{v_range}{default_string}: ')
    try:
        i = input()
        if isinstance(i, str) and not len(i):
            v = default
            print(f'{v}')
        else:
            v = literal_eval(i)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        v = None
    except:
        print('Unexpected error')
        raise
    if not is_num(v, v_min, v_max):
        print('Illegal input, enter a valid number')
        v = input_num(s, v_min, v_max, default)
    return v

def input_int_list(s=None, v_min=None, v_max=None):
    if v_min is not None and not isinstance(v_min, int):
        illegal_value(vmin, 'vmin', 'input_int_list') 
        return None
    if v_max is not None:
        if not isinstance(v_max, int):
            illegal_value(vmax, 'vmax', 'input_int_list') 
            return None
        if v_max < v_min:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return None
    if v_min is not None and v_max is not None:
        v_range = f' (each value in ({v_min}, {v_max}))'
    elif v_min is not None:
        v_range = f' (each value >= {v_min})'
    elif v_max is not None:
        v_range = f' (each value <= {v_max})'
    else:
        v_range = ''
    if s is None:
        print(f'Enter a series of integers{v_range}: ')
    else:
        print(f'{s}{v_range}: ')
    try:
        l = string_to_list(input())
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        l = None
    except:
        print('Unexpected error')
        raise
    if (not isinstance(l, list) or
            True in [True if not is_int(v, v_min, v_max) else False for v in l]):
        print('Illegal input: enter a valid set of dash/comma/whitespace separated integers '+
                'e.g. 2,3,5-8,10')
        l = input_int_list(s, v_min, v_max)
    return l

def input_yesno(s=None, default=None):
    if default is not None:
        if not isinstance(default, str):
            illegal_value(default, 'default', 'input_yesno') 
            return None
        if default.lower() in 'yes':
            default = 'y'
        elif default.lower() in 'no':
            default = 'n'
        else:
            illegal_value(default, 'default', 'input_yesno') 
            return None
        default_string = f' [{default}]'
    else:
        default_string = ''
    if s is None:
        print(f'Enter yes or no{default_string}: ')
    else:
        print(f'{s}{default_string}: ')
    i = input()
    if isinstance(i, str) and not len(i):
        i = default
        print(f'{i}')
    if i is not None and i.lower() in 'yes':
        v = True
    elif i is not None and i.lower() in 'no':
        v = False
    else:
        print('Illegal input, enter yes or no')
        v = input_yesno(s, default)
    return v

def input_menu(items, default=None, header=None):
    if not isinstance(items, (tuple, list)) or False in [True if isinstance(i, str) else False
            for i in items]:
        illegal_value(items, 'items', 'input_menu') 
        return None
    if default is not None:
        if not (isinstance(default, str) and default in items):
            logging.error(f'Illegal value for default ({default}), must be in {items}') 
            return None
        default_string = f' [{items.index(default)+1}]'
    else:
        default_string = ''
    if header is None:
        print(f'Choose one of the following items (1, {len(items)}){default_string}:')
    else:
        print(f'{header} (1, {len(items)}){default_string}:')
    for i, choice in enumerate(items):
        print(f'  {i+1}: {choice}')
    try:
        choice  = input()
        if isinstance(choice, str) and not len(choice):
            choice = items.index(default)
            print(f'{choice+1}')
        else:
            choice = literal_eval(choice)
            if isinstance(choice, int) and 1 <= choice <= len(items):
                choice -= 1
            else:
                raise ValueError
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        choice = None
    except:
        print('Unexpected error')
        raise
    if choice is None:
        print(f'Illegal choice, enter a number between 1 and {len(items)}')
        choice = input_menu(items, default)
    return choice

def create_mask(x, bounds=None, reverse_mask=False, current_mask=None):
    # bounds is a pair of number in the same units a x
    if not isinstance(x, (tuple, list, np.ndarray)) or not len(x):
        logging.warning(f'Illegal input array ({x}, {type(x)})')
        return None
    if bounds is not None and not is_num_pair(bounds):
        logging.warning(f'Illegal bounds parameter ({bounds} {type(bounds)}, input ignored')
        bounds = None
    if bounds is not None:
        if not reverse_mask:
            mask = np.logical_and(x > min(bounds), x < max(bounds))
        else:
            mask = np.logical_or(x < min(bounds), x > max(bounds))
    else:
        mask = np.ones(len(x), dtype=bool)
    if current_mask is not None:
        if not isinstance(current_mask, (tuple, list, np.ndarray)) or len(current_mask) != len(x):
            logging.warning(f'Illegal current_mask ({current_mask}, {type(current_mask)}), '+
                    'input ignored')
        else:
            mask = np.logical_and(mask, current_mask)
    if not True in mask:
        logging.warning('Entire data array is masked')
    return mask

def draw_mask_1d(ydata, xdata=None, current_index_ranges=None, current_mask=None,
        select_mask=True, num_index_ranges_max=None, title=None, legend=None, test_mode=False):
    def draw_selections(ax):
        ax.clear()
        ax.set_title(title)
        ax.legend([legend])
        ax.plot(xdata, ydata, 'k')
        for (low, upp) in current_include:
            xlow = 0.5*(xdata[max(0, low-1)]+xdata[low])
            xupp = 0.5*(xdata[upp]+xdata[min(num_data-1, upp+1)])
            ax.axvspan(xlow, xupp, facecolor='green', alpha=0.5)
        for (low, upp) in current_exclude:
            xlow = 0.5*(xdata[max(0, low-1)]+xdata[low])
            xupp = 0.5*(xdata[upp]+xdata[min(num_data-1, upp+1)])
            ax.axvspan(xlow, xupp, facecolor='red', alpha=0.5)
        for (low, upp) in selected_index_ranges:
            xlow = 0.5*(xdata[max(0, low-1)]+xdata[low])
            xupp = 0.5*(xdata[upp]+xdata[min(num_data-1, upp+1)])
            ax.axvspan(xlow, xupp, facecolor=selection_color, alpha=0.5)
        ax.get_figure().canvas.draw()

    def onclick(event):
        if event.inaxes in [fig.axes[0]]:
            selected_index_ranges.append(index_nearest_upp(xdata, event.xdata))

    def onrelease(event):
        if len(selected_index_ranges) > 0:
            if isinstance(selected_index_ranges[-1], int):
                if event.inaxes in [fig.axes[0]]:
                    event.xdata = index_nearest_low(xdata, event.xdata)
                    if selected_index_ranges[-1] <= event.xdata:
                        selected_index_ranges[-1] = (selected_index_ranges[-1], event.xdata)
                    else:
                        selected_index_ranges[-1] = (event.xdata, selected_index_ranges[-1])
                    draw_selections(event.inaxes)
                else:
                    selected_index_ranges.pop(-1)

    def confirm_selection(event):
        plt.close()

    def clear_last_selection(event):
        if len(selected_index_ranges):
            selected_index_ranges.pop(-1)
        draw_selections(ax)

    def update_mask(mask):
        for (low, upp) in selected_index_ranges:
            selected_mask = np.logical_and(xdata >= xdata[low], xdata <= xdata[upp])
            mask = np.logical_or(mask, selected_mask)
        for (low, upp) in unselected_index_ranges:
            unselected_mask = np.logical_and(xdata >= xdata[low], xdata <= xdata[upp])
            mask[unselected_mask] = False
        return mask

    def update_index_ranges(mask):
        # Update the currently included index ranges (where mask is True)
        current_include = []
        for i, m in enumerate(mask):
            if m == True:
                if len(current_include) == 0 or type(current_include[-1]) == tuple:
                    current_include.append(i)
            else:
                if len(current_include) > 0 and isinstance(current_include[-1], int):
                    current_include[-1] = (current_include[-1], i-1)
        if len(current_include) > 0 and isinstance(current_include[-1], int):
            current_include[-1] = (current_include[-1], num_data-1)
        return current_include

    # Check for valid inputs
    ydata = np.asarray(ydata)
    if ydata.ndim > 1:
        logging.warning(f'Illegal ydata dimension ({ydata.ndim})')
        return None, None
    num_data = ydata.size
    if xdata is None:
        xdata = np.arange(num_data)
    else:
        xdata = np.asarray(xdata, dtype=np.float64)
        if xdata.ndim > 1 or xdata.size != num_data:
            logging.warning(f'Illegal xdata shape ({xdata.shape})')
            return None, None
        if not np.all(xdata[:-1] < xdata[1:]):
            logging.warning('Illegal xdata: must be monotonically increasing')
            return None, None
    if current_index_ranges is not None:
        if not isinstance(current_index_ranges, (tuple, list)):
            logging.warning('Illegal current_index_ranges parameter ({current_index_ranges}, '+
                    f'{type(current_index_ranges)})')
            return None, None
    if not isinstance(select_mask, bool):
        logging.warning('Illegal select_mask parameter ({select_mask}, {type(select_mask)})')
        return None, None
    if num_index_ranges_max is not None:
        logging.warning('num_index_ranges_max input not yet implemented in draw_mask_1d')
    if title is None:
        title = 'select ranges of data'
    elif not isinstance(title, str):
        illegal(title, 'title')
        title = ''
    if legend is None and not isinstance(title, str):
        illegal(legend, 'legend')
        legend = None

    if select_mask:
        title = f'Click and drag to {title} you wish to include'
        selection_color = 'green'
    else:
        title = f'Click and drag to {title} you wish to exclude'
        selection_color = 'red'

    # Set initial selected mask and the selected/unselected index ranges as needed
    selected_index_ranges = []
    unselected_index_ranges = []
    selected_mask = np.full(xdata.shape, False, dtype=bool)
    if current_index_ranges is None:
        if current_mask is None:
            if not select_mask:
                selected_index_ranges = [(0, num_data-1)]
                selected_mask = np.full(xdata.shape, True, dtype=bool)
        else:
            selected_mask = np.copy(np.asarray(current_mask, dtype=bool))
    if current_index_ranges is not None and len(current_index_ranges):
        current_index_ranges = sorted([(low, upp) for (low, upp) in current_index_ranges])
        for (low, upp) in current_index_ranges:
            if low > upp or low >= num_data or upp < 0:
                continue
            if low < 0:
                low = 0
            if upp >= num_data:
                upp = num_data-1
            selected_index_ranges.append((low, upp))
        selected_mask = update_mask(selected_mask)
    if current_index_ranges is not None and current_mask is not None:
        selected_mask = np.logical_and(current_mask, selected_mask)
    if current_mask is not None:
        selected_index_ranges = update_index_ranges(selected_mask)

    # Set up range selections for display
    current_include = selected_index_ranges
    current_exclude = []
    selected_index_ranges = []
    if not len(current_include):
        if select_mask:
            current_exclude = [(0, num_data-1)]
        else:
            current_include = [(0, num_data-1)]
    else:
        if current_include[0][0] > 0:
            current_exclude.append((0, current_include[0][0]-1))
        for i in range(1, len(current_include)):
            current_exclude.append((current_include[i-1][1]+1, current_include[i][0]-1))
        if current_include[-1][1] < num_data-1:
            current_exclude.append((current_include[-1][1]+1, num_data-1))

    if not test_mode:

        # Set up matplotlib figure
        plt.close('all')
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        draw_selections(ax)

        # Set up event handling for click-and-drag range selection
        cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
        cid_release = fig.canvas.mpl_connect('button_release_event', onrelease)

        # Set up confirm / clear range selection buttons
        confirm_b = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
        clear_b = Button(plt.axes([0.59, 0.05, 0.15, 0.075]), 'Clear')
        cid_confirm = confirm_b.on_clicked(confirm_selection)
        cid_clear = clear_b.on_clicked(clear_last_selection)

        # Show figure
        plt.show(block=True)

        # Disconnect callbacks when figure is closed
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_release)
        confirm_b.disconnect(cid_confirm)
        clear_b.disconnect(cid_clear)

    # Swap selection depending on select_mask
    if not select_mask:
        selected_index_ranges, unselected_index_ranges = unselected_index_ranges, \
                selected_index_ranges

    # Update the mask with the currently selected/unselected x-ranges
    selected_mask = update_mask(selected_mask)

    # Update the currently included index ranges (where mask is True)
    current_include = update_index_ranges(selected_mask)

    return selected_mask, current_include

def findImageFiles(path, filetype, name=None):
    if isinstance(name, str):
        name = f' {name} '
    else:
        name = ' '
    # Find available index range
    if filetype == 'tif':
        if not isinstance(path, str) or not os.path.isdir(path):
            illegal_value(path, 'path', 'findImageRange')
            return -1, 0, []
        indexRegex = re.compile(r'\d+')
        # At this point only tiffs
        files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and
                f.endswith('.tif') and indexRegex.search(f)])
        num_imgs = len(files)
        if num_imgs < 1:
            logging.warning('No available'+name+'files')
            return -1, 0, []
        first_index = indexRegex.search(files[0]).group()
        last_index = indexRegex.search(files[-1]).group()
        if first_index is None or last_index is None:
            logging.error('Unable to find correctly indexed'+name+'images')
            return -1, 0, []
        first_index = int(first_index)
        last_index = int(last_index)
        if num_imgs != last_index-first_index+1:
            logging.error('Non-consecutive set of indices for'+name+'images')
            return -1, 0, []
        paths = [os.path.join(path, f) for f in files]
    elif filetype == 'h5':
        if not isinstance(path, str) or not os.path.isfile(path):
            illegal_value(path, 'path', 'findImageRange')
            return -1, 0, []
        # At this point only h5 in alamo2 detector style
        first_index = 0
        with h5py.File(path, 'r') as f:
            num_imgs = f['entry/instrument/detector/data'].shape[0]
            last_index = num_imgs-1
        paths = [path]
    else:
        illegal_value(filetype, 'filetype', 'findImageRange')
        return -1, 0, []
    logging.debug('\nNumber of available'+name+f'images: {num_imgs}')
    logging.debug('Index range of available'+name+f'images: [{first_index}, '+
            f'{last_index}]')

    return first_index, num_imgs, paths

def selectImageRange(first_index, offset, num_imgs, name=None, num_required=None):
    if isinstance(name, str):
        name = f' {name} '
    else:
        name = ' '
    # Check existing values
    use_input = False
    if (is_int(first_index, 0) and is_int(offset, 0) and is_int(num_imgs, 1)):
        if offset < 0:
            use_input = input_yesno(f'\nCurrent{name}first index = {first_index}, '+
                    'use this value (y/n)?', 'y')
        else:
            use_input = input_yesno(f'\nCurrent{name}first index/offset = '+
                    f'{first_index}/{offset}, use these values (y/n)?', 'y')
        if num_required is None:
            if use_input:
                use_input = input_yesno(f'Current number of{name}images = '+
                        f'{num_imgs}, use this value (y/n)? ', 'y')
    if use_input:
        return first_index, offset, num_imgs

    # Check range against requirements
    if num_imgs < 1:
        logging.warning('No available'+name+'images')
        return -1, -1, 0
    if num_required is None:
        if num_imgs == 1:
            return first_index, 0, 1
    else:
        if not is_int(num_required, 1):
            illegal_value(num_required, 'num_required', 'selectImageRange')
            return -1, -1, 0
        if num_imgs < num_required:
            logging.error('Unable to find the required'+name+
                    f'images ({num_imgs} out of {num_required})')
            return -1, -1, 0

    # Select index range
    print('\nThe number of available'+name+f'images is {num_imgs}')
    if num_required is None:
        last_index = first_index+num_imgs
        use_all = f'Use all ([{first_index}, {last_index}])'
        pick_offset = 'Pick a first index offset and a number of images'
        pick_bounds = 'Pick the first and last index'
        choice = input_menu([use_all, pick_offset, pick_bounds], default=pick_offset)
        if not choice:
            offset = 0
        elif choice == 1:
            offset = input_int('Enter the first index offset', 0, last_index-first_index)
            first_index += offset
            if first_index == last_index:
                num_imgs = 1
            else:
                num_imgs = input_int('Enter the number of images', 1, num_imgs-offset)
        else:
            offset = input_int('Enter the first index', first_index, last_index)
            first_index += offset
            num_imgs = input_int('Enter the last index', first_index, last_index)-first_index+1
    else:
        use_all = f'Use ([{first_index}, {first_index+num_required-1}])'
        pick_offset = 'Pick the first index offset'
        choice = input_menu([use_all, pick_offset], pick_offset)
        offset = 0
        if choice == 1:
            offset = input_int('Enter the first index offset', 0, num_imgs-num_required)
            first_index += offset
        num_imgs = num_required

    return first_index, offset, num_imgs

def loadImage(f, img_x_bounds=None, img_y_bounds=None):
    """Load a single image from file.
    """
    if not os.path.isfile(f):
        logging.error(f'Unable to load {f}')
        return None
    img_read = plt.imread(f)
    if not img_x_bounds:
        img_x_bounds = (0, img_read.shape[0])
    else:
        if (not isinstance(img_x_bounds, (tuple, list)) or len(img_x_bounds) != 2 or 
                not (0 <= img_x_bounds[0] < img_x_bounds[1] <= img_read.shape[0])):
            logging.error(f'inconsistent row dimension in {f}')
            return None
    if not img_y_bounds:
        img_y_bounds = (0, img_read.shape[1])
    else:
        if (not isinstance(img_y_bounds, list) or len(img_y_bounds) != 2 or 
                not (0 <= img_y_bounds[0] < img_y_bounds[1] <= img_read.shape[1])):
            logging.error(f'inconsistent column dimension in {f}')
            return None
    return img_read[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]

def loadImageStack(files, filetype, img_offset, num_imgs, num_img_skip=0,
        img_x_bounds=None, img_y_bounds=None):
    """Load a set of images and return them as a stack.
    """
    logging.debug(f'img_offset = {img_offset}')
    logging.debug(f'num_imgs = {num_imgs}')
    logging.debug(f'num_img_skip = {num_img_skip}')
    logging.debug(f'\nfiles:\n{files}\n')
    img_stack = np.array([])
    if filetype == 'tif':
        img_read_stack = []
        i = 1
        t0 = time()
        for f in files[img_offset:img_offset+num_imgs:num_img_skip+1]:
            if not i%20:
                logging.info(f'    loading {i}/{num_imgs}: {f}')
            else:
                logging.debug(f'    loading {i}/{num_imgs}: {f}')
            img_read = loadImage(f, img_x_bounds, img_y_bounds)
            img_read_stack.append(img_read)
            i += num_img_skip+1
        img_stack = np.stack([img_read for img_read in img_read_stack])
        logging.info(f'... done in {time()-t0:.2f} seconds!')
        logging.debug(f'img_stack shape = {np.shape(img_stack)}')
        del img_read_stack, img_read
    elif filetype == 'h5':
        if not isinstance(files[0], str) and not os.path.isfile(files[0]):
            illegal_value(files[0], 'files[0]', 'loadImageStack')
            return img_stack
        t0 = time()
        logging.info(f'Loading {files[0]}')
        with h5py.File(files[0], 'r') as f:
            shape = f['entry/instrument/detector/data'].shape
            if len(shape) != 3:
                logging.error(f'inconsistent dimensions in {files[0]}')
            if not img_x_bounds:
                img_x_bounds = (0, shape[1])
            else:
                if (not isinstance(img_x_bounds, (tuple, list)) or len(img_x_bounds) != 2 or 
                        not (0 <= img_x_bounds[0] < img_x_bounds[1] <= shape[1])):
                    logging.error(f'inconsistent row dimension in {files[0]} {img_x_bounds} '+
                            f'{shape[1]}')
            if not img_y_bounds:
                img_y_bounds = (0, shape[2])
            else:
                if (not isinstance(img_y_bounds, list) or len(img_y_bounds) != 2 or 
                        not (0 <= img_y_bounds[0] < img_y_bounds[1] <= shape[2])):
                    logging.error(f'inconsistent column dimension in {files[0]}')
            img_stack = f.get('entry/instrument/detector/data')[
                    img_offset:img_offset+num_imgs:num_img_skip+1,
                    img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
        logging.info(f'... done in {time()-t0:.2f} seconds!')
    else:
        illegal_value(filetype, 'filetype', 'loadImageStack')
    return img_stack

def combine_tiffs_in_h5(files, num_imgs, h5_filename):
    img_stack = loadImageStack(files, 'tif', 0, num_imgs)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('entry/instrument/detector/data', data=img_stack)
    del img_stack
    return [h5_filename]

def clearImshow(title=None):
    plt.ioff()
    if title is None:
        title = 'quick imshow'
    elif not isinstance(title, str):
        illegal_value(title, 'title', 'clearImshow')
        return
    plt.close(fig=title)

def clearPlot(title=None):
    plt.ioff()
    if title is None:
        title = 'quick plot'
    elif not isinstance(title, str):
        illegal_value(title, 'title', 'clearPlot')
        return
    plt.close(fig=title)

def quickImshow(a, title=None, path=None, name=None, save_fig=False, save_only=False,
            clear=True, extent=None, show_grid=False, grid_color='w', grid_linewidth=1, **kwargs):
    if title is not None and not isinstance(title, str):
        illegal_value(title, 'title', 'quickImshow')
        return
    if path is not None and not isinstance(path, str):
        illegal_value(path, 'path', 'quickImshow')
        return
    if not isinstance(save_fig, bool):
        illegal_value(save_fig, 'save_fig', 'quickImshow')
        return
    if not isinstance(save_only, bool):
        illegal_value(save_only, 'save_only', 'quickImshow')
        return
    if not isinstance(clear, bool):
        illegal_value(clear, 'clear', 'quickImshow')
        return
    if not title:
        title='quick imshow'
#    else:
#        title = re.sub(r"\s+", '_', title)
    if name is None:
        ttitle = re.sub(r"\s+", '_', title)
        if path is None:
            path = f'{ttitle}.png'
        else:
            path = f'{path}/{ttitle}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if extent is None:
        extent = (0, a.shape[1], a.shape[0], 0)
    if clear:
        plt.close(fig=title)
    if not save_only:
        plt.ion()
    plt.figure(title)
    plt.imshow(a, extent=extent, **kwargs)
    if show_grid:
        ax = plt.gca()
        ax.grid(color=grid_color, linewidth=grid_linewidth)
#    if title != 'quick imshow':
#        plt.title = title
    if save_only:
        plt.savefig(path)
        plt.close(fig=title)
    else:
        if save_fig:
            plt.savefig(path)

def quickPlot(*args, xerr=None, yerr=None, vlines=None, title=None, xlim=None, ylim=None,
        xlabel=None, ylabel=None, legend=None, path=None, name=None, show_grid=False, 
        save_fig=False, save_only=False, clear=True, block=False, **kwargs):
    if title is not None and not isinstance(title, str):
        illegal_value(title, 'title', 'quickPlot')
        title = None
    if xlim is not None and not isinstance(xlim, (tuple, list)) and len(xlim) != 2:
        illegal_value(xlim, 'xlim', 'quickPlot')
        xlim = None
    if ylim is not None and not isinstance(ylim, (tuple, list)) and len(ylim) != 2:
        illegal_value(ylim, 'ylim', 'quickPlot')
        ylim = None
    if xlabel is not None and not isinstance(xlabel, str):
        illegal_value(xlabel, 'xlabel', 'quickPlot')
        xlabel = None
    if ylabel is not None and not isinstance(ylabel, str):
        illegal_value(ylabel, 'ylabel', 'quickPlot')
        ylabel = None
    if legend is not None and not isinstance(legend, (tuple, list)):
        illegal_value(legend, 'legend', 'quickPlot')
        legend = None
    if path is not None and not isinstance(path, str):
        illegal_value(path, 'path', 'quickPlot')
        return
    if not isinstance(show_grid, bool):
        illegal_value(show_grid, 'show_grid', 'quickPlot')
        return
    if not isinstance(save_fig, bool):
        illegal_value(save_fig, 'save_fig', 'quickPlot')
        return
    if not isinstance(save_only, bool):
        illegal_value(save_only, 'save_only', 'quickPlot')
        return
    if not isinstance(clear, bool):
        illegal_value(clear, 'clear', 'quickPlot')
        return
    if not isinstance(block, bool):
        illegal_value(block, 'block', 'quickPlot')
        return
    if title is None:
        title = 'quick plot'
#    else:
#        title = re.sub(r"\s+", '_', title)
    if name is None:
        ttitle = re.sub(r"\s+", '_', title)
        if path is None:
            path = f'{ttitle}.png'
        else:
            path = f'{path}/{ttitle}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if clear:
        plt.close(fig=title)
    args = unwrap_tuple(args)
    if depth_tuple(args) > 1 and (xerr is not None or yerr is not None):
        logging.warning('Error bars ignored form multiple curves')
    if not save_only:
        if block:
            plt.ioff()
        else:
            plt.ion()
    plt.figure(title)
    if depth_tuple(args) > 1:
       for y in args:
           plt.plot(*y, **kwargs)
    else:
        if xerr is None and yerr is None:
            plt.plot(*args, **kwargs)
        else:
            plt.errorbar(*args, xerr=xerr, yerr=yerr, **kwargs)
    if vlines is not None:
        for v in vlines:
            plt.axvline(v, color='r', linestyle='--', **kwargs)
#    if vlines is not None:
#        for s in tuple(([x, x], list(plt.gca().get_ylim())) for x in vlines):
#            plt.plot(*s, color='red', **kwargs)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if show_grid:
        ax = plt.gca()
        ax.grid(color='k')#, linewidth=1)
    if legend is not None:
        plt.legend(legend)
    if save_only:
        plt.savefig(path)
        plt.close(fig=title)
    else:
        if save_fig:
            plt.savefig(path)
        if block:
            plt.show(block=block)

def selectArrayBounds(a, x_low=None, x_upp=None, num_x_min=None, ask_bounds=False,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a numpy array.
    """
    if isinstance(a, (tuple, list)):
        a = np.array(a)
    if not isinstance(a, np.ndarray) or a.ndim != 1:
        illegal_value(a.ndim, 'array type or dimension', 'selectArrayBounds')
        return None
    len_a = len(a)
    if num_x_min is None:
        num_x_min = 1
    else:
        if num_x_min < 2 or num_x_min > len_a:
            logging.warning('Illegal value for num_x_min in selectArrayBounds, input ignored')
            num_x_min = 1

    # Ask to use current bounds
    if ask_bounds and (x_low is not None or x_upp is not None):
        if x_low is None:
            x_low = 0
        if not is_int(x_low, 0, len_a-num_x_min):
            illegal_value(x_low, 'x_low', 'selectArrayBounds')
            return None
        if x_upp is None:
            x_upp = len_a
        if not is_int(x_upp, x_low+num_x_min, len_a):
            illegal_value(x_upp, 'x_upp', 'selectArrayBounds')
            return None
        quickPlot((range(len_a), a), vlines=(x_low,x_upp), title=title)
        if not input_yesno(f'\nCurrent array bounds: [{x_low}, {x_upp}] '+
                    'use these values (y/n)?', 'y'):
            x_low = None
            x_upp = None
        else:
            clearPlot(title)
            return x_low, x_upp

    if x_low is None:
        x_min = 0
        x_max = len_a
        x_low_max = len_a-num_x_min
        while True:
            quickPlot(range(x_min, x_max), a[x_min:x_max], title=title)
            zoom_flag = input_yesno('Set lower data bound (y) or zoom in (n)?', 'y')
            if zoom_flag:
                x_low = input_int('    Set lower data bound', 0, x_low_max)
                break
            else:
                x_min = input_int('    Set lower zoom index', 0, x_low_max)
                x_max = input_int('    Set upper zoom index', x_min+1, x_low_max+1)
    else:
        if not is_int(x_low, 0, len_a-num_x_min):
            illegal_value(x_low, 'x_low', 'selectArrayBounds')
            return None
    if x_upp is None:
        x_min = x_low+num_x_min
        x_max = len_a
        x_upp_min = x_min
        while True:
            quickPlot(range(x_min, x_max), a[x_min:x_max], title=title)
            zoom_flag = input_yesno('Set upper data bound (y) or zoom in (n)?', 'y')
            if zoom_flag:
                x_upp = input_int('    Set upper data bound', x_upp_min, len_a)
                break
            else:
                x_min = input_int('    Set upper zoom index', x_upp_min, len_a-1)
                x_max = input_int('    Set upper zoom index', x_min+1, len_a)
    else:
        if not is_int(x_upp, x_low+num_x_min, len_a):
            illegal_value(x_upp, 'x_upp', 'selectArrayBounds')
            return None
    print(f'lower bound = {x_low} (inclusive)\nupper bound = {x_upp} (exclusive)]')
    quickPlot((range(len_a), a), vlines=(x_low,x_upp), title=title)
    if not input_yesno('Accept these bounds (y/n)?', 'y'):
        x_low, x_upp = selectArrayBounds(a, None, None, num_x_min, title=title)
    clearPlot(title)
    return x_low, x_upp

def selectImageBounds(a, axis, low=None, upp=None, num_min=None,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a 2D numpy array.
    """
    if isinstance(a, np.ndarray):
        if a.ndim != 2:
            illegal_value(a.ndim, 'array dimension', 'selectImageBounds')
            return None
    elif isinstance(a, (tuple, list)):
        if len(a) != 2:
            illegal_value(len(a), 'array dimension', 'selectImageBounds')
            return None
        if len(a[0]) != len(a[1]) or not (isinstance(a[0], (tuple, list, np.ndarray)) and
                isinstance(a[1], (tuple, list, np.ndarray))):
            logging.error(f'Illegal array type in selectImageBounds ({type(a[0])} {type(a[1])})')
            return None
        a = np.array(a)
    else:
        illegal_value(a, 'array type', 'selectImageBounds')
        return None
    if axis < 0 or axis >= a.ndim:
        illegal_value(axis, 'axis', 'selectImageBounds')
        return None
    low_save = low
    upp_save = upp
    num_min_save = num_min
    if num_min is None:
        num_min = 1
    else:
        if num_min < 2 or num_min > a.shape[axis]:
            logging.warning('Illegal input for num_min in selectImageBounds, input ignored')
            num_min = 1
    if low is None:
        min_ = 0
        max_ = a.shape[axis]
        low_max = a.shape[axis]-num_min
        while True:
            if axis:
                quickImshow(a[:,min_:max_], title=title, aspect='auto',
                        extent=[min_,max_,a.shape[0],0])
            else:
                quickImshow(a[min_:max_,:], title=title, aspect='auto',
                        extent=[0,a.shape[1], max_,min_])
            zoom_flag = input_yesno('Set lower data bound (y) or zoom in (n)?', 'y')
            if zoom_flag:
                low = input_int('    Set lower data bound', 0, low_max)
                break
            else:
                min_ = input_int('    Set lower zoom index', 0, low_max)
                max_ = input_int('    Set upper zoom index', min_+1, low_max+1)
    else:
        if not is_int(low, 0, a.shape[axis]-num_min):
            illegal_value(low, 'low', 'selectImageBounds')
            return None
    if upp is None:
        min_ = low+num_min
        max_ = a.shape[axis]
        upp_min = min_
        while True:
            if axis:
                quickImshow(a[:,min_:max_], title=title, aspect='auto',
                        extent=[min_,max_,a.shape[0],0])
            else:
                quickImshow(a[min_:max_,:], title=title, aspect='auto',
                        extent=[0,a.shape[1], max_,min_])
            zoom_flag = input_yesno('Set upper data bound (y) or zoom in (n)?', 'y')
            if zoom_flag:
                upp = input_int('    Set upper data bound', upp_min, a.shape[axis])
                break
            else:
                min_ = input_int('    Set upper zoom index', upp_min, a.shape[axis]-1)
                max_ = input_int('    Set upper zoom index', min_+1, a.shape[axis])
    else:
        if not is_int(upp, low+num_min, a.shape[axis]):
            illegal_value(upp, 'upp', 'selectImageBounds')
            return None
    bounds = (low, upp)
    a_tmp = np.copy(a)
    a_tmp_max = a.max()
    if axis:
        a_tmp[:,bounds[0]] = a_tmp_max
        a_tmp[:,bounds[1]-1] = a_tmp_max
    else:
        a_tmp[bounds[0],:] = a_tmp_max
        a_tmp[bounds[1]-1,:] = a_tmp_max
    print(f'lower bound = {low} (inclusive)\nupper bound = {upp} (exclusive)')
    quickImshow(a_tmp, title=title)
    del a_tmp
    if not input_yesno('Accept these bounds (y/n)?', 'y'):
        bounds = selectImageBounds(a, axis, low=low_save, upp=upp_save, num_min=num_min_save,
            title=title)
    return bounds


class Config:
    """Base class for processing a config file or dictionary.
    """
    def __init__(self, config_file=None, config_dict=None):
        self.config = {}
        self.load_flag = False
        self.suffix = None

        # Load config file 
        if config_file is not None and config_dict is not None:
            logging.warning('Ignoring config_dict (both config_file and config_dict are specified)')
        if config_file is not None:
           self.loadFile(config_file)
        elif config_dict is not None:
           self.loadDict(config_dict)

    def loadFile(self, config_file):
        """Load a config file.
        """
        if self.load_flag:
            logging.warning('Overwriting any previously loaded config file')
        self.config = {}

        # Ensure config file exists
        if not os.path.isfile(config_file):
            logging.error(f'Unable to load {config_file}')
            return

        # Load config file (for now for Galaxy, allow .dat extension)
        self.suffix = os.path.splitext(config_file)[1]
        if self.suffix == '.yml' or self.suffix == '.yaml' or self.suffix == '.dat':
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        elif self.suffix == '.txt':
            with open(config_file, 'r') as f:
                lines = f.read().splitlines()
            self.config = {item[0].strip():literal_eval(item[1].strip()) for item in
                    [line.split('#')[0].split('=') for line in lines if '=' in line.split('#')[0]]}
        else:
            illegal_value(self.suffix, 'config file extension', 'Config.loadFile')

        # Make sure config file was correctly loaded
        if isinstance(self.config, dict):
            self.load_flag = True
        else:
            logging.error(f'Unable to load dictionary from config file: {config_file}')
            self.config = {}

    def loadDict(self, config_dict):
        """Takes a dictionary and places it into self.config.
        """
        if self.load_flag:
            logging.warning('Overwriting the previously loaded config file')

        if isinstance(config_dict, dict):
            self.config = config_dict
            self.load_flag = True
        else:
            illegal_value(config_dict, 'dictionary config object', 'Config.loadDict')
            self.config = {}

    def saveFile(self, config_file):
        """Save the config file (as a yaml file only right now).
        """
        suffix = os.path.splitext(config_file)[1]
        if suffix != '.yml' and suffix != '.yaml':
            illegal_value(suffix, 'config file extension', 'Config.saveFile')

        # Check if config file exists
        if os.path.isfile(config_file):
            logging.info(f'Updating {config_file}')
        else:
            logging.info(f'Saving {config_file}')

        # Save config file
        with open(config_file, 'w') as f:
            yaml.safe_dump(self.config, f)

    def validate(self, pars_required, pars_missing=None):
        """Returns False if any required keys are missing.
        """
        if not self.load_flag:
            logging.error('Load a config file prior to calling Config.validate')

        def validate_nested_pars(config, par):
            par_levels = par.split(':')
            first_level_par = par_levels[0]
            try:
                first_level_par = int(first_level_par)
            except:
                pass
            try:
                next_level_config = config[first_level_par]
                if len(par_levels) > 1:
                    next_level_par = ':'.join(par_levels[1:])
                    return validate_nested_pars(next_level_config, next_level_par)
                else:
                    return True
            except:
                return False

        pars_missing = [p for p in pars_required if not validate_nested_pars(self.config, p)]
        if len(pars_missing) > 0:
            logging.error(f'Missing item(s) in configuration: {", ".join(pars_missing)}')
            return False
        else:
            return True
