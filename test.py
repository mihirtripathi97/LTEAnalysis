from pv_analyzer import pv_analyze
import numpy as np

def emp_kep(v_rot, v_100):
        
        r_au = 100.*np.power(v_100/v_rot, 2)

        return r_au

pv_b6 = pv_analyze(pv_path='uid___A002_Xb5ee5a_X288a.ms.split.cal.l1489irs.spw3.cvel_chavg2.cube.clean_2_6_8_05.image.pbcor.regridded.smoothened.PV_69_w1.fits',
                    is_Tb=True)

pv_b6.read_PV(print_details = False)

Tb_df = pv_b6.get_tb_on_curve(curve_function = emp_kep, get_surrounding_pix = True,
                              num_pix = 3, cf_kwargs = {'v_100':3.2})

print(Tb_df)