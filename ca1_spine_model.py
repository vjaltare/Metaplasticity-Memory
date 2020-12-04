import numpy as np 
import sys
from scipy.integrate import odeint

class ca1_spine:
	"""Computational biophysical model of CA1 spine"""
		#### Global constants:
	rtol = 1e-6
	atol = 1e-6 
	F = 96485.33 ## Coulomb/mole
	Nav = 6.022e23
	e = 2.718
	############## Defining constant parameters ##################
	## Spine compartment and ER size:
	Vspine = 0.06 ## um^3
	d_spine = (6*Vspine/3.14)**0.333  ## um
	Aspine = 3.14 * d_spine**2 * 1e-8  ## cm^2
	Vspine = Vspine * 1e-15 ## liter
	Aer = 0.1 * Aspine  ## cm^2
	## Parameters for endogenous immobile buffer (CBP): 
	kbuff_f = 247 ## /uM/s
	kbuff_b = 524 ## /s

	## Parameters for endogenous slow buffer:
	kslow_f = 24.7 ## /uM/s
	kslow_b = 52.4 ## /s

	## Parameters for calbindin-Ca2+ kinetics:
	km0m1=174 ## /uM/s
	km1m2=87 ## /uM/s
	km1m0=35.8 ## /s
	km2m1=71.6 ## /s
	kh0h1=22 ## /uM/s
	kh1h2=11 ## /uM/s
	kh1h0=2.6 ## /s
	kh2h1=5.2 ## /s

	## Parameters for PMCA and NCX pumps:
	k1H,k2H,k3H,kH_leak = [150,15,12,3.33]  ## (/uM/s, /s, /s, /s)
	k1L,k2L,k3L,kL_leak = [300,300,600,10]  ## (/uM/s, /s, /s, /s)

	## Parameters for CaM-Ca2+ interaction:
	k1c_on = 6.8  ## /uM/s
	k1c_off = 68  ## /s
	k2c_on = 6.8 ## /uM/s
	k2c_off = 10 ## /s
	k1n_on = 108 ## /uM/s
	k1n_off = 4150 ## /s
	k2n_on = 108 ## /uM/s
	k2n_off = 800 ## /s

	## Membrane and leak parameters:
	Cmem = 1e-6 ##  F/cm^2
	g_L = 2e-4  ## S/cm^2
	E_L = -70   ## mV

	## AMPA receptor parameters:
	tau_A1 = 0.2e-3 ## s
	tau_A2 = 2e-3  ## s
	E_A = 0  ## mV
	g_A = 0.5e-9  ## S

	## NMDA receptor parameters:
	tau_N1 = 5e-3 ## s
	tau_N2 = 50e-3 ## s
	E_N = 0  ## mV
	g_N = 65
	# insert NMDA conductance #

		## L-VGCC parameters:
	um = -20 ## mV
	kmv = 5  ## mV
	tau_mv = 0.08e-3 ## sec
	uh = -65  ## mV
	khv = -7 ## mV			 
	tau_hv = 300e-3  ## sec

	## Spine neck parameters:
	Rneck = 1e8  ## Ohm
	gc = 1.0/Rneck ## S
	rho_spines = 0  ## Surface density of co-active synaptic inputs on dendritic compartment (cm^-2)

	## SERCA kinetic parameters:
	Vmax_serca = 1  ## uM/sec
	Kd_serca = 0.2 ## uM

	## Parameters for Ca2+-based plasticity model:
	P1,P2,P3,P4 = [1.0,10.0,0.001,2]
	beta1,beta2 = [60,60]  ## /uM
	alpha1,alpha2 = [2.0,20.0] ## uM

	######################### ER Buffer Calreticulin Parameters ####################################
	#### Site C ####
	Kdiss_c = 2e3 #uM
	Scer = 20 #mol Ca/mol
	kcerf = 0.1 #uM^-1 s^-1

	#### Site P ####
	Kdiss_p = 10 #uM
	Sp = 1 #mol Ca/mol
	kpf = 100 #uM^-1 s^-1

	kcerb = kcerf * Kdiss_c
	kpb = kpf * Kdiss_p 

	################# ER Buffer ###########################
	totBer_0 = 3.6e3 #uM
	totBcer = Scer * totBer_0 #uM
	totBper = Sp * totBer_0 #uM
	buff_flag = 1

	###############################################################################################
	############################################ SOCC Model ##########################################
	############################################################### Defining parameters ######################################################
	V_socc_max = 1.57 #um/s per channel max flux
	K_socc = 50 #um/s
	tau_socc = 30 #s
	n_socc = 4 #Hill coefficient
	socc_clus = 10 # no of socc channels
	###########################################################################################################################################



	#########################################################
	########### Concentrations of various species ###########
	#########################################################

	## External Ca (uM):
	ca_ext = 2e3

	## Resting cytosolic Ca (uM):ip3
	ca_0 = 0.05

	## Resting Ca in ER (uM):
	ca_er_0 = 260#1.21E4#1.182e4#250 

	## Total calbindin concentration in spine (uM):
	calb_tot = 45

	## Total CBP concentration in the spine (uM):
	cbp_tot = 80

	## Total slow buffer concentration in the spine (uM):
	Bslow_tot = 40

	## Total concentration of PMCA and NCX pumps in the spine head (uM):
	pHtot = (1e14) * 1000 * Aspine/(Nav * Vspine)
	pLtot = (1e14) * 140 * Aspine/(Nav * Vspine)

	## Total concentration of CaM in spine (uM):
	cam_tot = 50


	#########################################################################################
	################ RyR ########################################
	#### The model consists of a gating scheme consisting of 5 closed states and 2 open states ####
	#################### Rate Constants ###################################
	kon = 712.0 #1/(uM.s)
	koff = 3000.0 #### units of all the quantities hereafter is s^-1 ######
	kc4o1 = 10000.0 
	kc4o2 = 1
	ko1c4 = 500
	ko2c4 = 0.5
	ko1c5 = 2.0
	ko2c5 = 3000.0
	kc5o1 = 0.6666
	kc5o2 = 100.0
	kc5I  = 0.5
	kIc5 = 1.5
	kRc1 = 4*kon
	kc1R = koff
	kc1c2 = 3*kon
	kc2c1 = 2*koff
	kc2c3 = 2*kon
	kc3c2 = 3*koff
	kc3c4 = kon
	kc4c3 = 4*koff
	##########################################################
	######## RyR parameters (play around these to make sure everyting works well together) ##########
	gRyR = 18.5 #s^-1  ## Johenning's data -> see analysis
	####################################################################################
	################### SK Channel parameters ##########################################
	tau_sk = 6.3e-3 #s
	n_sk = 4
	g_sk = 25e-12 #S
	E_sk = -90 #mV
	###################################################################################
	######################################################################################
	g_N_Ca = 0.1 * (g_N/(2*F*78.0*ca_ext)) * 1e6   ## Ca conductance of NMDAR channels; liter/sec
	k_erleak = Vmax_serca * (ca_0**2)/((Kd_serca**2 + ca_0**2)*(ca_er_0 - ca_0))  ## /s
	g_vgcc = g_N_Ca
	############################################################################


	def __init__(self, g_N, nRyR, tau_refill, k_sk, Vr):
		"""initialize the spine. parameters: (nmda conductance, #ryr, tau_refill, SK schannel affinity parameter, Ver/Vspine)"""
		self.g_N = float(g_N) * 1e-12 #S
		self.nRyR = int(nRyR)
		self.tau_refill = float(tau_refill)
		self.Vr = float(Vr)
		self.k_sk = float(k_sk)

		self.Ver = self.Vr * self.Vspine  ## liter
		self.Vspine = self.Vspine-self.Ver  ## liter


##############################################################################################
#### Temporal profile of glutamate availability:
	def glu(self, t, flag):
	 	"""temporal profile of glutamate availability"""
	 	if flag == 0:
	 		return 0
	 	if flag > 0:	 		
		 	self.tau_glu = 1e-3 #sec
		 	self.glu_0 = 2.718 * 300 #um

		 	total = 0

		 	for tpuff in self.tpre:
		 		if t > tpuff: total += self.glu_0 * np.exp(-(t-tpuff)/self.tau_glu) * ((t-tpuff)/self.tau_glu)
		 	return total

	##############################################################################################

	##############################################################################################
	#### Voltage profile of BAP at the dendritic compartment:
	##Use this to get the voltage of the spine whenever required
	def u_bpap(self, t):

		V0 = 67
		total = 0

		for tbp in tpost:
			if t > tbp: total += V0 * (0.7 * np.exp(-(t - tbp)/0.003) + 0.3 * np.exp(-(t - tbp)/0.04))
		return self.E_L + total

	##############################################################################################    
	#### AMPAR conductance profile: 
	def I_A(self,flag,u,t):

	    if flag==0:
	        return 0
	    else:
	        total = 0
	        for tpuff in tpre:
	            if t>tpuff: total += self.g_A * (np.exp(-(t-tpuff)/self.tau_A2) - np.exp(-(t-tpuff)/self.tau_A1))  
	        return total * (u - self.E_A)


	def I_N(self, flag, u, t):
		"""use this function to get the NMDA receptor current"""

		if flag == 0:
			return 0

		else:
			total = 0
			for tpuff in tpre:
				if t > tpuff : total+= self.g_N * (np.exp(-(t - tpuff)/self.tau_N2) - np.exp(-(t - tpuff)/self.tau_N1))
			return total * (u - self.E_N)/(1 + 0.28 * np.exp(-0.062 * u))

	#############################################################################################################33
	############### SK Channel Hill function ###########################################
	def sk_inf(self, x):
		"""SK-Hill function. args: cytosolic_calcium """
		return np.power(x, self.n_sk)/(np.power(x, self.n_sk) + np.power(self.k_sk, self.n_sk))


	##############################################################################################        
	#### Plasticity model, Omega function

	def omega(self, x):
		"""defines the Omega function to classify LTP/LTD based on induction var x"""
		U = -self.beta2 * (x - self.alpha2)
		V = -self.beta1 * (x - self.alpha1)

		if U > 100: U = 100
		if V > 100: V = 100
		return (1./(1. + np.exp(U))) - 0.5*(1./(1 + np.exp(V)))

	#############################################################################################
	#### Plasticity model Omega function	

	def omega_tau(self, x):
		"""defines the tau function for synaptic weight change"""
		return self.P1 + (self.P2/(self.P3 + (2 * x /(self.alpha1 + self.alpha2))**self.P4))

	#######################################################################
	##### Coupled ODEs describing ER bearing spine head ###################
	def spine_model(self,x,t):
	    
	    
	    pH, pL, cbp, Bslow, calb, calb_m1, calb_h1, calb_m2, calb_h2, calb_m1h1, calb_m2h1, calb_m1h2, c1n0, c2n0, c0n1, c0n2, c1n1, c2n1, c1n2, c2n2, mv, hv, w, u, ud,\
	    ca_er, ca, R, c1, c2, c3, c4, c5, o1, o2, Bcer, Bper, p_socc, s_sk = x
	    ###########|----------- RyR ----------|   |-Buffer-|  |-socc-||-sk-|##############

	    nt = self.glu(t,self.flag)
	    	

	    if (self.flag>0 and input_pattern=='stdp'): 
	        ud = self.u_bpap(t)
	        
	    
	    ca_eq = 0.01
	    
	    p_socc_inf = np.power(self.K_socc,self.n_socc)/(np.power(self.K_socc,self.n_socc) + np.power(ca_er, self.n_socc))
	    dPdt = (p_socc_inf - p_socc)/self.tau_socc


	    
	    ## RyR Kinetics:#################################################
	    Jrel = self.gRyR  * (o1 + o2) * (ca_er - ca) 
	    ######################### Channel Ca Release ########################################
	    ca_eq += self.nRyR * Jrel * self.Vr + self.V_socc_max * self.socc_clus * p_socc 
	    ca_er_eq = - self.nRyR * Jrel  + (self.ca_er_0 - ca_er)/self.tau_refill
	########################################################################################################################
	    
	    ## Buffer equations:

	    Bslow_eq = -self.kslow_f*Bslow*ca + self.kslow_b*(self.Bslow_tot - Bslow)
	    ca_eq += -self.kslow_f*Bslow*ca + self.kslow_b*(self.Bslow_tot - Bslow)
	    
	    cbp_eq = -self.kbuff_f*ca*cbp + self.kbuff_b*(self.cbp_tot - cbp)
	    ca_eq += -self.kbuff_f*ca*cbp + self.kbuff_b*(self.cbp_tot - cbp)    
	    
	    calb_m2h2 = self.calb_tot - calb - calb_m1 - calb_h1 - calb_m2 - calb_h2 - calb_m1h1 - calb_m2h1 - calb_m1h2
	    calb_eqs = [ -ca*calb*(self.km0m1 + self.kh0h1) + self.km1m0*calb_m1 + self.kh1h0*calb_h1,\
	                     ca*calb*self.km0m1 - self.km1m0*calb_m1 + calb_m2*self.km2m1 - ca*calb_m1*self.km1m2 + calb_m1h1*self.kh1h0 - ca*calb_m1*self.kh0h1,\
	                     ca*calb*self.kh0h1 - self.kh1h0*calb_h1 + calb_h2*self.kh2h1 - ca*calb_h1*self.kh1h2 + calb_m1h1*self.km1m0 - ca*calb_h1*self.km0m1,\
	                     ca*calb_m1*self.km1m2 - self.km2m1*calb_m2 + self.kh1h0*calb_m2h1 - ca*self.kh0h1*calb_m2,\
	                     ca*calb_h1*self.kh1h2 - self.kh2h1*calb_h2 + self.km1m0*calb_m1h2 - ca*self.km0m1*calb_h2,\
	                     ca*(calb_h1*self.km0m1 + calb_m1*self.kh0h1) - (self.km1m0+self.kh1h0)*calb_m1h1 - ca*calb_m1h1*(self.km1m2+self.kh1h2) + self.kh2h1*calb_m1h2 + self.km2m1*calb_m2h1,\
	                     ca*self.km1m2*calb_m1h1 - self.km2m1*calb_m2h1 + self.kh2h1*calb_m2h2 - self.kh1h2*ca*calb_m2h1 + self.kh0h1*ca*calb_m2 - self.kh1h0*calb_m2h1,\
	                     ca*self.kh1h2*calb_m1h1 - self.kh2h1*calb_m1h2 + self.km2m1*calb_m2h2 - self.km1m2*ca*calb_m1h2 + self.km0m1*ca*calb_h2 - self.km1m0*calb_m1h2 ]
	    ca_eq += -ca*(self.km0m1*(calb+calb_h1+calb_h2) + self.kh0h1*(calb+calb_m1+calb_m2) + self.km1m2*(calb_m1+calb_m1h1+calb_m1h2) + self.kh1h2*(calb_h1+calb_m1h1+calb_m2h1))+\
	                self.km1m0*(calb_m1+calb_m1h1+calb_m1h2) + self.kh1h0*(calb_h1+calb_m1h1+calb_m2h1) + self.km2m1*(calb_m2+calb_m2h1+calb_m2h2) + self.kh2h1*(calb_h2+calb_m1h2+calb_m2h2)
	    #print(calb)
	    ## Ca2+/calmodulin kinetics:
	    
	    c0n0 = self.cam_tot - c1n0 - c2n0 - c0n1 - c0n2 - c1n1 - c2n1 - c1n2 - c2n2
	    c1n0_eq = -(self.k2c_on*ca + self.k1c_off + self.k1n_on*ca)*c1n0 + self.k1c_on*ca*c0n0 + self.k2c_off*c2n0 + self.k1n_off*c1n1
	    c2n0_eq = -(self.k2c_off + self.k1n_on*ca)*c2n0 + self.k2c_on*ca*c1n0 + self.k1n_off*c2n1
	    c0n1_eq = -(self.k2n_on*ca + self.k1n_off + self.k1c_on*ca)*c0n1 + self.k1n_on*ca*c0n0 + self.k2n_off*c0n2 + self.k1c_off*c1n1
	    c0n2_eq = -(self.k2n_off + self.k1c_on*ca)*c0n2 + self.k2n_on*ca*c0n1 + self.k1c_off*c1n2
	    c1n1_eq = -(self.k2c_on*ca + self.k1c_off + self.k1n_off + self.k2n_on*ca)*c1n1 + self.k1c_on*ca*c0n1 + self.k1n_on*ca*c1n0 + self.k2c_off*c2n1 + self.k2n_off*c1n2
	    c2n1_eq = -(self.k2c_off + self.k2n_on*ca)*c2n1 + self.k2c_on*ca*c1n1 + self.k2n_off*c2n2 + self.k1n_on*ca*c2n0 - self.k1n_off*c2n1
	    c1n2_eq = -(self.k2n_off + self.k2c_on*ca)*c1n2 + self.k2n_on*ca*c1n1 + self.k2c_off*c2n2 + self.k1c_on*ca*c0n2 - self.k1c_off*c1n2
	    c2n2_eq = -(self.k2c_off + self.k2n_off)*c2n2 + self.k2c_on*ca*c1n2 + self.k2n_on*ca*c2n1
	    cam_eqs = [c1n0_eq, c2n0_eq, c0n1_eq, c0n2_eq, c1n1_eq, c2n1_eq, c1n2_eq, c2n2_eq]
	    ca_eq += -ca*(self.k1c_on*(c0n0+c0n1+c0n2) + self.k1n_on*(c0n0+c1n0+c2n0) + self.k2c_on*(c1n0+c1n1+c1n2) + self.k2n_on*(c0n1+c1n1+c2n1)) + \
	    self.k1c_off*(c1n0+c1n1+c1n2) + self.k1n_off*(c0n1+c1n1+c2n1) + self.k2c_off*(c2n0+c2n1+c2n2) + self.k2n_off*(c0n2+c1n2+c2n2)
	    
	    ############################ ER Buffer kinetics #############################################################
	    if self.buff_flag:
	        dBcer_dt = -self.kcerf*(ca_er * Bcer) + self.kcerb * (self.totBcer - Bcer)
	        dBper_dt = -self.kpf * (ca_er * Bper) + self.kpb * (self.totBper - Bper)
	        ca_er_eq += -self.kcerf*(ca_er * Bcer) + self.kcerb * (self.totBcer - Bcer) -self.kpf * (ca_er * Bper) + self.kpb * (self.totBper - Bper)
	    else:
	        dBcer_dt = 0
	        dBper_dt = 0
	        ca_er_eq += 0
	 
	    ## PMCA/NCX kinetics:
	    
	    ca_eq += pH*self.kH_leak - ca*pH*self.k1H + self.k2H*(self.pHtot - pH)  +  pL*self.kL_leak - ca*pL*self.k1L + self.k2L*(self.pLtot - pL)
	    pH_eq = self.k3H*(self.pHtot - pH) - ca*pH*self.k1H + self.k2H*(self.pHtot - pH)
	    pL_eq = self.k3L*(self.pLtot - pL) - ca*pL*self.k1L +self.k2L*(self.pLtot - pL)
	    
	      ## SERCA kinetics: 							

	    ca_eq += -self.Vmax_serca * ca**2/(self.Kd_serca**2 + ca**2) + self.k_erleak*(ca_er - ca)

	    ## VGCC equatiosn:

	    mv_eq = ((1.0/(1 + np.exp(-(u-self.um)/self.kmv))) - mv)/self.tau_mv
	    hv_eq = ((1.0/(1 + np.exp(-(u-self.uh)/self.khv))) - hv)/self.tau_hv
	    I_vgcc = -0.001 * self.Nav * 3.2e-19 * self.g_vgcc * (mv**2) * hv * 0.078 * u * (ca - self.ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u))

	    ## Spine and dendrite voltage eqns:
	    

	    sp_hh_eq = -(1/self.Cmem) * ( self.g_L*(u - self.E_L) + self.I_A(self.flag,u,t)/self.Aspine + self.I_N(self.flag,u,t)/self.Aspine - (self.gc/self.Aspine)*(ud - u) - I_vgcc/self.Aspine - self.g_sk * s_sk * (u - self.E_sk))
	    dend_hh_eq = -(1/self.Cmem) * ( self.g_L*(ud - self.E_L) + self.rho_spines*self.gc*(ud - u))

	    ## Ca2+ influx through NMDAR and VGCC:

	    ca_eq += -(self.g_N_Ca/self.Vspine) * (self.I_N(self.flag,u,t)/(self.g_N*(u - self.E_N))) * 0.078 * u * (ca - self.ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u)) \
	            -(self.g_vgcc/self.Vspine) * (mv**2) * hv * 0.078 * u * (ca - self.ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u))   
	    
	    ################################## RyR Dynamics #################################################
	    # dRdt  = -R*ca*kRc1 + c1*kc1R
	    # dc1dt = -(kc1c2*ca+ kc1R)*c1 + c2*kc2c1 + kRc1*ca*R
	    # dc2dt = -(kc2c3*ca + kc2c1)*c2 + kc3c2*c3 + kc1c2*ca*c1
	    # dc3dt = -(kc3c4*ca + kc3c2)*c3 + kc2c3*ca*c2 + kc4c3*c4
	    # dc4dt = -(kc4o1 + kc4o2 + kc4c3)*c4 + kc3c4*ca*c3 + ko1c4*o1 + ko2c4*o2
	    # dc5dt = -(kc5o1 + kc5o2 + kc5I)*c5 + ko1c5*o1 + kIc5*(1-R-c1-c2-c3-c4-c5-o1-o2) + ko2c5*o2
	    # do1dt = -(ko1c5 + ko1c4)*o1 + kc4o1*c4 + kc5o1*c5
	    # do2dt = -(ko2c4 + ko2c5)*o2 + kc4o2*c4 + kc5o2*c5
	    ryr_eqs = [-R*ca*self.kRc1 + c1*self.kc1R, -(self.kc1c2*ca+ self.kc1R)*c1 + c2*self.kc2c1 + self.kRc1*ca*R, -(self.kc2c3*ca + self.kc2c1)*c2 + self.kc3c2*c3 + self.kc1c2*ca*c1,\
	    			-(self.kc3c4*ca + self.kc3c2)*c3 + self.kc2c3*ca*c2 + self.kc4c3*c4, -(self.kc4o1 + self.kc4o2 + self.kc4c3)*c4 + self.kc3c4*ca*c3 + self.ko1c4*o1 + self.ko2c4*o2,\
	    			-(self.kc5o1 + self.kc5o2 + self.kc5I)*c5 + self.ko1c5*o1 + self.kIc5*(1-R-c1-c2-c3-c4-c5-o1-o2) + self.ko2c5*o2, -(self.ko1c5 + self.ko1c4)*o1 + self.kc4o1*c4 + self.kc5o1*c5,\
	    			 -(self.ko2c4 + self.ko2c5)*o2 + self.kc4o2*c4 + self.kc5o2*c5]
	    ##################################################################################################

	    ########################## SK - channel equations ################################################
	    dskdt = (self.sk_inf(ca) - s_sk)/self.tau_sk 

	    ## Equation for plasticity variable w:
	    acam = self.cam_tot - c0n0    
	    w_eq = (1.0/self.omega_tau(acam))*(self.omega(acam) - w)
	    
	    #print('') [dRdt, dc1dt, dc2dt, dc3dt, dc4dt, dc5dt, do1dt, do2dt]
	    print([pH_eq, pL_eq, cbp_eq, Bslow_eq] + calb_eqs + cam_eqs + [mv_eq, hv_eq] + [w_eq] + [sp_hh_eq, dend_hh_eq,\
	            ca_er_eq, ca_eq] + ryr_eqs + [dBcer_dt, dBper_dt] + [dPdt] + [dskdt])

	    return [pH_eq, pL_eq, cbp_eq, Bslow_eq] + calb_eqs + cam_eqs + [mv_eq, hv_eq] + [w_eq] + [sp_hh_eq, dend_hh_eq,\
	            ca_er_eq, ca_eq] + ryr_eqs + [dBcer_dt, dBper_dt] + [dPdt] + [dskdt]

#################################################################################################################
################################################ Experiments ##################################################
#################### simulate ER+ spine without inputs to cnverge to steady state #############
	def get_resting_params(self, flag):
		self.flag = flag
		self.buff_flag = 0
		##########################################################################################################
		######################## Initializing all variables:######################################################
		########################################################################################################


		pumps_init = [self.pHtot, self.pLtot]
		buff_init =  [self.cbp_tot, self.Bslow_tot] + [self.calb_tot,0,0,0,0,0,0,0] 
		CaM_init = [0]*8  
		vgcc_init = [0,1] 
		w_init = [0] 
		voltage_init = [self.E_L, self.E_L]
		#ip3r_init = [1]
		ca_init = [self.ca_er_0, self.ca_0]
		RyR_init = [1/8]*8
		calreticulin_init = [63840,135]
		SOCC_init = [0.001]
		sk_init = [0.1]
		     
		xinit0 = pumps_init + buff_init + CaM_init + vgcc_init + w_init + voltage_init + ca_init + RyR_init + calreticulin_init + SOCC_init + sk_init

		print(xinit0)

		################ solving #################################################
		t0 = np.arange(0., 100., 0.001)

		sol = odeint(self.spine_model, xinit0, t0, rtol=self.rtol, atol=self.atol)
		return sol[-1,:]


######################################### RDP ################################################################

	def do_rdp(self, f_input, n_inputs):
		"""perform RDP. parameters: (frequency of stimulation, no. of presynaptic inputs)"""
		self.tpre = [float(i/f_input) for i in range(n_inputs)]
		t = np.arange(0., n_inputs/f_input, 0.001)






