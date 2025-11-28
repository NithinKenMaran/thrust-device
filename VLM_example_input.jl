
include("VLM_solver.jl")
using .LinearRotorVLM, StaticArrays

# --- 1. Geometry and operating point (input data about the propeller blades here) --

Rtip = 0.065                # tip radius [m]
rhat = [0.309,0.3544,0.3998,0.4452,0.4906,0.536,0.5814,0.6268,0.6722,0.7176,0.763,
        0.8084,0.8538,0.8992,0.9446,1.0]  
        # normalised (r/Rtip) values, these are the points where the blade is 
        #sectioned (for the grid generation), it should monotonically increase
        #from the hub to tip ratio and end in 1.0

chord = [0.17920886224,0.17450678352,0.16906135312,0.16414993488,0.16614102544,
         0.17531555504,0.18487896528,0.1934313816,0.20106841648,0.207885076,
         0.2139716464,0.2194114784,0.22428003488,0.22864473376,0.23256525904,
         0.23614584048] # these are the normalized chord (c/Rtip) values at 
         #each of those r/Rtip locations, the function will convert using the
         #Rtip values given above

pitch_deg = [67.0166650531887,63.5593269991508,60.3496487360009,57.3799857767955,
             54.6386285284852,52.1114023135303,49.782910038437,47.6374384318261,
             45.6595798306243,43.8346303097947,42.148821370796,40.5894333670659,
             39.1448284966006,37.8044316323507,36.5586793193302,35.3989510816643] 
             # these are the pitch angle (in degrees) from the plane of rotation 
             #of the rotor when the blade is twisted about the leading edge. 
             #these are also given at the locations of r/Rtip given before
             #positive pitch is 

sections = LinearRotorVLM.make_sections(rhat, chord, pitch_deg, Rtip) 
# calls the function to section the blade

hub_center = @SVector [0.0, 0.0, 0.0] 
#this is the center of rotation of the rotor, it is to be kept as 
#this 0 vector itself
psi = 0.0 
# the azimuthal angle of the blade, this is to be kept as 0 to just find the 
#thrust and torque

nc = 8 
# nc = number of chord sections
# chord is sectioned using cosine spacing.
# number of span sections is derived from provided r/Rtip values
# clearly, number of panels made = number of span sections x number of chord sections


blade = LinearRotorVLM.build_blade(sections, hub_center, psi; nc=nc) 
# assembles all the panels on the blade

B    = 11 # number of blades on the rotor
Vinf = @SVector [0.0, 0.0, -50.69]          
# axial inflow along -z (coming to the rotor from +z), negative
#flow means it is flowing into the rotor

Omega = @SVector [0.0, 0.0, 1464.924654]    
# rad/s, counterclockwise positive rotation about +z

rho = 1.225 # density of air

op = LinearRotorVLM.OperatingPoint(Vinf, Omega, rho) # inputs operating conditions

rotor = LinearRotorVLM.make_rotor(blade, B, hub_center, Rtip) 
#assembles the rotor out of blades

# --- 2. One assemble call to set wakes on panels ---

A, b, panels, wake_endpoints =
    LinearRotorVLM.assemble_system(blade, op; wake_length = 4.0)  
    # sets wakes as described in the math section

# --- 3. Airfoil polar (replace with your real data) ---

#the following make a table, with each list as a column, they tell the code the
#Cl and Cd for the airfoil of the propeller at each angle of attack, the index 
#number is the row, this is just normal data about the airfoil that is
#taken from online sources like XFOIL or airfoiltools.com

alpha_tab = [-9.5,-9.25,-9,-8.75,-8.5,-8.25,-8,-7.75,-7.5,-7.25,-7,-6.75,-6.5,-6.25,
-6,-5.75,-5.5,-5.25,-5,-4.75,-4.5,-4.25,-4,-3.75,-3.5,-3.25,-3,-2.75,-2.5,-2.25,-2,
-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,
2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,
7.75,8,8.25,8.5,8.75,9,9.25,9.5,9.75,10,10.25,10.5,10.75,11,11.25,11.5,11.75,
12,12.25,12.5,12.75,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.25,
15.5,15.75,16,16.25,16.5,16.75,17]      # angle of attack in degrees
cl_tab    = [-0.3426,-0.3784,-0.4173,-0.3682,-0.3611,-0.3724,-0.4032,-0.4436,
-0.4819,-0.4471,-0.448,-0.4588,-0.4847,-0.4783,-0.4744,-0.4612,-0.4309,
-0.3801,-0.339,-0.2986,-0.2547,-0.2155,-0.1732,-0.1342,-0.0916,-0.0546,
-0.0135,0.0233,0.0621,0.0998,0.1378,0.1759,0.2137,0.2473,0.3011,0.3304,
0.3765,0.4028,0.4335,0.472,0.4988,0.5348,0.5631,0.5902,0.6265,0.6506,0.6774,
0.711,0.7347,0.7611,0.7929,0.8161,0.8423,0.8728,0.8957,0.9217,0.9513,0.9739,
0.9995,1.0286,1.0508,1.0755,1.1016,1.1279,1.1534,1.1757,1.1987,1.2212,1.2409,
1.2594,1.2769,1.2933,1.3086,1.3197,1.3281,1.3316,1.3276,1.3204,1.3127,1.3077,
1.3074,1.311,1.3189,1.3289,1.3439,1.3595,1.3787,1.3912,1.4202,1.4239,1.4319,
1.4604,1.454,1.4507,1.4512,1.4783,1.4597,1.4404,1.4201,1.3983,1.3742,1.3474,
1.3171,1.2836,1.2473,1.2101,1.1753]      # corresponding corefficient of lift
                    #at those same aoa values (same index means same aoa)

cd_tab = [0.10705,0.10671,0.10641,0.09949,0.09726,0.09561,0.09481,0.09403,0.09082,
0.0883,0.0864,0.08412,0.07929,0.07718,0.07516,0.06965,0.0673,0.03672,0.03524,
0.03253,0.03033,0.02874,0.02773,0.02656,0.02579,0.02511,0.02457,0.02414,
0.02363,0.02306,0.02248,0.02179,0.02102,0.02001,0.01877,0.01877,0.01838,
0.01841,0.01835,0.01803,0.01808,0.01782,0.01782,0.01789,0.01763,0.01784,
0.01798,0.01786,0.01814,0.01836,0.01834,0.0187,0.01895,0.01902,0.01941,
0.0197,0.01982,0.02025,0.02057,0.02074,0.02119,0.02148,0.02159,0.02168,
0.02184,0.02207,0.02216,0.02223,0.0224,0.02262,0.02285,0.02308,0.02333,
0.02372,0.02421,0.02495,0.02605,0.02765,0.02965,0.03176,0.03378,0.03571,
0.03755,0.03922,0.04092,0.0425,0.0443,0.04604,0.04801,0.05002,0.05205,
0.05447,0.05696,0.05963,0.06218,0.06519,0.0685,0.07234,0.07667,0.0815,
0.0869,0.09302,0.10006,0.10833,0.11795,0.12886,0.14068] 

#similar to cl_tab but for coefficient of drag


alpha_tab, cl_tab, cd_tab = LinearRotorVLM.extend_polar_viterna(
    alpha_tab, cl_tab, cd_tab; 
    alpha_stall_deg = 13.5, #the angle of attack in degrees at which the 
                            #airfoil of propeller stalls
    cd_max          = 0.14068, #max Cd of the airfoil
    alpha_max_deg   = 150.0) #this is the new range of the 
                            #data after expansion

# --- 4. Nonlinear VLM + polars solve for gamma ---

gamma = zeros(Float64, length(blade.panels))

gamma = LinearRotorVLM.solve_nonlinear_vlm_with_polars!(
    gamma, blade, op,
    rhat, chord, pitch_deg, Rtip, wake_endpoints;
    alpha_tab = alpha_tab,
    cl_tab    = cl_tab,
    nc        = nc,
    max_iter  = 30, # this limits the number of iterations in the 
                    #angle of attack loop
    relax     = 0.3, # under‑relaxation factor for stability, reduce if 
                     #the solver oscillates, start 
                    # at 0.1 and lower it if gamma/thrust oscillates between
                    #iteration count at low advance ratio ( Vinf/ (omega * D) )
)

# --- 5. Thrust from the vortex lattice with this gamma ---

thrust, torque = LinearRotorVLM.compute_thrust(blade, op, gamma; B = B)

thrust, torque

#The thrust is in Newton and the torque is negative of the torque
#to be provided by the motor
