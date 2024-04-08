from MCMC_estimator import pv_analyze


pv_b6 = pv_analyze(pv_path='uid___A002_Xb5ee5a_X288a.ms.split.cal.l1489irs.spw3.cvel_chavg2.cube.clean_2_6_8_05.image.pbcor.regridded.smoothened.PV_69_w1.fits',
                    is_Tb=True)

pv_b6.read_PV(print_details = True)

pv_b6.get_Tb_on_curve()

