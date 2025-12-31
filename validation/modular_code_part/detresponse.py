import numpy as np

def convlo(lo):
    """
    change the origin of the longitude 
    """
    lo=np.array([lo])
    loc=np.where(lo>180,lo-360,np.where(lo<-180,360+lo,lo))
    return loc[0]

def RIeff(ra, dec, psi, Aplus, Across, phic, GMST, uI, vI, xI, f):
    """
    Effective response as a function of extrinsic parameters
    
    ra: right ascension in uit of radian
    dec: declination in unit of radian
    psi： in unit of radian (note that we include an additional minus sign for psi 
          in comparison to 1809.10727 to match with Bilby injection)
    Aplus, Across: amplitudes of polorization in the source frame
    phic: coalescence phase in unit of radian
    GMST: the GMST coordinate derived from gps time and (ra, dec), in unit of hour
    uI, vI, xI: detector properties (xI in unit of meter)
    f: frequency
    """
    lo = convlo((ra*24/(2*np.pi)-GMST)*360/24)
    la = dec*180/np.pi
    theta0 = (90 - la)*np.pi/180
    phi0 = lo*np.pi/180
    nh =np.array([np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)])
    lh =np.array([np.cos(theta0)*np.cos(phi0),np.cos(theta0)*np.sin(phi0),-np.sin(theta0)])
    mh =np.array([-np.sin(phi0),np.cos(phi0),0])
    DeIplus=((uI.dot(lh))**2-(vI.dot(lh))**2-(uI.dot(mh))**2+(vI.dot(mh))**2)/2
    DeIcross=(uI.dot(lh))*(uI.dot(mh))-(vI.dot(lh))*(vI.dot(mh))
    A1=Aplus*np.cos(2*psi)*np.cos(2*phic)-Across*np.sin(-2*psi)*np.sin(2*phic)
    A2=Aplus*np.sin(-2*psi)*np.cos(2*phic)+Across*np.cos(2*psi)*np.sin(2*phic)
    A3=-Across*np.sin(-2*psi)*np.cos(2*phic)-Aplus*np.cos(2*psi)*np.sin(2*phic)
    A4=Across*np.cos(2*psi)*np.cos(2*phic)-Aplus*np.sin(-2*psi)*np.sin(2*phic)
    hplus=A1+1j*A3
    hcross=A2+1j*A4
    Reff=(DeIplus*hplus+DeIcross*hcross)*np.exp(2*1j*np.pi*f*nh.dot(xI)/(3*1e8))
    return Reff


def DtHL(ra, dec, GMST, xH, xL):
    """
    the signal arrival time lag tH-tL (positive if the signal arrives at L1 first) in unit of second
    ra: right ascension in uit of radian
    dec: declination in unit of radian
    GMST: the GMST coordinate derived from gps time and (ra, dec), in unit of hour
    xI: detector vertices in unit of meter
    """
    lo = convlo((ra*24/(2*np.pi)-GMST)*360/24)
    la = dec*180/np.pi
    theta0 = (90 - la)*np.pi/180
    phi0 = lo*np.pi/180
    nh =np.array([np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)])
    DtHL0 = -nh.dot(xH-xL)/(3*1e8)
    return DtHL0
