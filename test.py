
from typing import List
#duration: float/int = 
#unit: [str] hour , secs min

#input: 124, seconds, [hours, minutes, seconds]
#output: 0 hours, 2 minutes, 4 seconds

#[decades, years, mon]

# def seconds2hours(inval,outval)

map_units = {
    'hour': 3600, # factor convert  seconds [h * factor = seconds]
    'minutes': 60
    }


def convert_time(
    input_val:float, 
    curr_unit:str, 
    out_units:List[str],
    unit_convert_seconds_value:dict:Dic={}
    ) -> List[int]:

    ## 1 check it using defualt unit_convert_seconds_value or default
        ## 1.2 if user defined check it it valid unit_convert_seconds_value

    ## 2. Create empty out_units
    ## iterate over the list to make the ovetions

    ## 3. Convert input to senconds `input_in_senconds`

    ## 4. iterate over dict tails units. convet `input_in_senconds` to the expected units 



    input_in_senconds = unit_convert_seconds_value[curr_unit] * input_val
    
    for u_curr in out_units:
        # convet curr value from second to `u_curr`
        temp_value = input_in_senconds / unit_convert_seconds_value[u_curr]
        out_units.append(temp_value)
        # convert to sends and  subtract remaning to get the ramaining 
        input_in_senconds = input_in_senconds - unit_convert_seconds_value[u_curr] * temp_value

    return out_units










