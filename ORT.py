# -*- coding: utf-8 -*-
"""
Optical Ray Tracer v4.1

Lukas Kostal, 9.12.2022, Imperial College London

This module provides functions and classes for a simple 3D optical ray tracer
which can be used to investigate geometrical optics phenomena in compound
optical systems comprised of flat and spherical surfaces.

Examples
--------
Examples can be found in the Examples folder submitted together with the module
or on the GitHub repository of the project.

Functions
---------
get_mag(vec)
    Calculate the magnitude of a vector.

get_norm(vec)
    Normalise a vector.

get_circ(r, n)
    Return x, y coordinates of points in a uniform circluar distribution.

get_sqr(r, n)
    Return x, y coordinates of points in a uniform square distribution.

get_hline(r, n)
    Return x, y coordinates of points in a uniform linear distribution.

snell(i, n, n1, n2, R)
    Apply Snell's law to find direction of propagation of refracted ray.

get_R(curv)
    Convert curvature into radius of curvature.

Classes
-------
Ray()
    Class to represent a single ray of light.

Beam()
    Class to represent a beam of light as a collection of rays.

PropElem()
    Base class for all optical elements which propagate light.

FlatSurf(PropElem)
    Class to represent a flat refracting surface.

SpherSurf(PropElem)
    Class to represent a spherical refracting surface.

SpherLens(PropElem)
    Class to represent a spherical lens.

Screen()
    Class to represent a screen on which light is imaged.

"""


import numpy as np
import matplotlib.pyplot as plt


class PhysicalError(Exception):
    """
    Exception for when an error caused by a physical phenomenon is encountered.
    """

    def __init__(self, msg):
        self.msg = msg


class ShapeError(Exception):
    """
    Exception for when an array with an incorrect shape is encountered.
    """

    def __init__(self, msg):
        self.msg = msg


def get_mag(vec):
    """
    Calculate the magnitude of a vector.

    Parameters
    ----------
    vec : list or numpy.ndarray
        1D array representing the input vector in Cartasean coordinates.

    Raises
    ------
    None

    Returns
    -------
    mag : numpy.float64
        Magnitude of the input vector.

    Raise
    -----
    None
    """

    return np.sqrt(np.dot(vec, vec))


def get_norm(vec):
    """
    Normalise a vector.

    Parameters
    ----------
    vec : list or numpy.ndarray
        1D array representing the input vector in Cartasean coordinates.

    Raises
    ------
    ValueError
        When the input vector has magnitude of 0 so can not be normalised.

    Returns
    -------
    vec : numpy.ndarray
        Normalised input vector.

    Rase
    ----
    None
    """
    if get_mag(vec) == 0:
        raise ValueError("Input vector must have a non-zero magnitude.")

    return vec / get_mag(vec)


def get_circ(r, n):
    """
    Return the x, y coordinates of points with a circular distribution with a
    uniform density.

    Parameters
    ----------
    r : float
        Radius of the circular distirubtion of points. Must be +ve.

    n : int
        Number of points per specified radius. Must be a +ve integer.

    Raises
    ------
    ValueError
        When the input parameters have inappropriate values.

    Returns
    -------
    x_grid : numpy.ndarray
        Array of x coordinates of the generated points.

    y_grid : numpy.ndarray
        Array of y coordinates of the generated points.
    """

    if r <= 0:
        raise ValueError("r must be a +ve value.")

    if n <= 0 or type(n) != int:
        raise ValueError("n must be a +ve integer.")

    r_arr = np.linspace(0, r, n)

    # find the number of radial points that fits at each radius in r_arr
    n_tht = (2 * np.pi * r_arr) // (r / (n - 1))
    n_tht[0] = 1
    n_tht = n_tht.astype(int)

    tht_grid = np.array([])

    # create an array of angles corresponding to each radius
    for i in range(0, n):
        tht_arr = np.linspace(0, 2 * np.pi, n_tht[i])
        tht_grid = np.append(tht_grid, tht_arr)

    # repeat points in r_arr to match the size of tht_arr
    r_grid = np.repeat(r_arr, n_tht)

    x_grid = r_grid * np.sin(tht_grid)
    y_grid = r_grid * np.cos(tht_grid)

    return x_grid, y_grid


def get_sqr(r, n):
    """
    Return the x, y coordinates of points with a square distribution with a
    uniform density.

    Parameters
    ----------
    r : float
        Side length of the square distirubtion of points. Must be +ve.

    n : int
        Number of points per side length. Must be a +ve integer.

    Raises
    ------
    ValueError
        When the input parameters have inappropriate values.

    Returns
    -------
    x_grid : numpy.ndarray
        Array of x coordinates of the generated points.

    y_grid : numpy.ndarray
        Array of y coordinates of the generated points.
    """

    if r <= 0:
        raise ValueError("r must be a +ve value.")

    if n <= 0 or type(n) != int:
        raise ValueError("n must be a +ve integer.")

    r_arr = np.linspace(-r/2, r/2, n)
    x_grid, y_grid = np.meshgrid(r_arr, r_arr)

    x_grid = np.concatenate(x_grid)
    y_grid = np.concatenate(y_grid)

    return x_grid, y_grid


def get_hline(r, n):
    """
    Generate the x, y coordinates of points distributed along a line
    parallel to the x axis.

    Parameters
    ----------
    r : float
        Length of the line on which the points are distributed. Must be +ve.

    n : int
        Number of points on the line. Must be a +ve integer.

    Raises
    ------
    ValueError
        When the input parameters have inappropriate values.

    Returns
    -------
    x_grid : numpy.ndarray
        Array of x coordinates of the generated points.

    y_grid : numpy.ndarray
        Array of y coordinates of the generated points.
    """

    if r <= 0:
        raise ValueError("r must be a +ve value.")

    if n <= 0 or type(n) != int:
        raise ValueError("n must be a +ve integer.")

    x_arr = np.linspace(-r, r, n)
    y_arr = np.repeat(0, n)

    return x_arr, y_arr


def snell(i, n, n1, n2):
    """
    Apply Snell's law to find direction of propagation of refracted ray.

    Parameters
    ----------
    i : numpy.ndarray or list
        Direction of propagation of incident ray

    n : numpy.ndarray or list
        Normal to the refracting surface

    n1 : float
         Refractive index of medium in which incident ray propagates

    n2 : float
         Refractive index of medium in which refracted ray propagates

    Raises
    ------
    PhysicalError
        When incident ray undergoes total internal reflection.

    Returns
    -------
    r : numpy.ndarray
        Direction of propagation of refracted ray
    """
    mu = n1 / n2

    to_sqrt = 1 - mu**2 * (1 - np.dot(n, i)**2)
    if to_sqrt < 0:
        raise PhysicalError("Total internal reflection occurs.")

    r = - np.sqrt(to_sqrt) * n + mu * (i - np.dot(n, i) * n)
    return r


def get_R(curv):
    """
    Convert curvature into radius of curvature.

    Parameters
    ----------
    curv : float
        Curvature to be converted into radius.

    Raises
    ------
    None

    Returns
    -------
    R : float or numpy.nan
        Calculated radius of curvature.

    Notes
    -----
    If curvature is 0 function returns numpy.nan.
    """

    if curv == 0:
        R = np.nan
    else:
        R = 1 / curv

    return R


class Ray():
    """
    The Ray class to represent a single ray of light.

    Attributes
    ----------
    _p : numpy.ndarray
        2D array of all position vectors (vertices) along axis 0.

    _k : numpy.ndarray
        2D array of direction of propagation vectors along axis 0 corresponding
        to the vertices in _p.

    _miss : bool
        Specify if ray does not intercept an optical element which it is
        propagated through.

    _color : string
        Color with which to plot the ray on the ray diagram.

    Methods
    -------
    __init__(self, p, k, color='red')
        Make an instance of the Ray class.

    __repr__(self)
        Representation method for the Ray class.

    __str__(self)
        String method for the Ray class.

    get_p(self)
        Return the current position vector of the ray.

    get_k(self)
        Return the current direction of propagation vector of the ray.

    get_vert(self)
        Return all of the position vectors (vertices) of the ray.

    append(self, p, k)
        Append a new position vector p and direction of propagation vector k
        to the ray.

    reset(self)
        Reset the ray by returning to the initial direction vector and the
        initial direction of propagation of the ray.

    show(self, miss=True, equal=False)
        Plot the ray on a 2D ray diagram using the specified color.
    """

    def __init__(self, p, k, color='red'):
        """
        Make an instance of the Ray class.

        Parameters
        ----------
        p : list or numpy.ndarray
            Initial position vector of the ray.

        k : list or numpy.ndarray
            Initial direction of propagation vector of the ray.

        color : string, optional
            Color with which to plot the ray. The default is 'red'.

        Raises
        ------
        ShapeError
            When the vectors used to initialise the ray have incorrect shapes.

        ValueError
            When the direction of propagation has zero magnitude.

        Returns
        -------
        None
        """

        p = np.array(p)
        k = np.array(k)

        if p.shape != (3,):
            raise ShapeError("Vector p has to be a 1D array of length 3.")

        if k.shape != (3,):
            raise ShapeError("Vector k has to be a 1D array of length 3.")

        if get_mag(k) == 0:
            raise ValueError("Direction of propagation vector must have a "
                             "non-zero magnitude.")

        k = get_norm(k)

        self._p = np.array([p])
        self._k = np.array([k])
        self._miss = False
        self._color = color

        return None


    def __repr__(self):
        """
        Representation method for the Ray class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which represents all of attributes of the Ray class
            listed in the order _p, _k, _miss, _color.
        """

        return "%s, %s, %s, %s" % (self._p, self._k, self._miss, self._color)


    def __str__(self):
        """
        String method for the Ray class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which shows all of attributes of the Ray class.
        """

        return "p = \n %s \n \n k = \n %s \n \n miss = %s \n color = %s" \
            % (self._p, self._k, self._miss, self._color)


    def get_p(self):
        """
        Return the current position vector of the ray

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        p : numpy.ndarray
            1D array representing the position vector.
        """

        return self._p[-1]


    def get_k(self):
        """
        Return the current normalised direction of propagation of the ray.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        k : numpy.ndarray
            1D array representing the normalised direction of propagation.
        """

        return self._k[-1]


    def get_vert(self):
        """
        Return all of the position vectors (vertices) of the ray.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        p : numpy.ndarray
            2D array of position vectors of the vertices along axis 0
        """

        return self._p


    def append(self, p, k):
        """
        Update the ray by appending a new position vector p and direction
        of propagation vector k.

        Parameters
        ----------
        p : numpy.ndarray or list
            New position vector to be appended.

        k : numpy.ndarray or list
            New direction of propagation vector to be appended.

        Raises
        ------
        DimensionalityError
            When the input parameters are not 1D arrays of length 3.

        ValueError
            When the direction of propagation has zero magnitude.

        Return
        ------
        None
        """

        p = np.array(p)
        k = np.array(k)

        if p.shape != (3,):
            raise ShapeError("Vector p has to be a 1D array of length 3.")

        if k.shape != (3,):
            raise ShapeError("Vector k has to be a 1D array of length 3.")

        if get_mag(k) == 0:
            raise ValueError("Direction of propagation vector must have a "
                             "non-zero magnitude.")

        k = get_norm(k)

        self._p = np.append(self._p, np.array([p]), axis=0)
        self._k = np.append(self._k, np.array([k]), axis=0)

        return None


    def reset(self):
        """
        Reset the ray by moving to the initial position vector and direction
        of propagation while removing all of the appended p and k.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Return
        ------
        None
        """

        self._p = np.array([self._p[0, :]])
        self._k = np.array([self._k[0, :]])
        self._miss = False

        return None


    def show(self, miss=True, equal=False):
        """
        Plot the ray on a 2D ray diagram using the specified color.

        Parameters
        ----------
        miss : bool, optional
            Specify whether to show a ray which does not intercept all of the
            optical elements. The default is True.

        equal : bool, optional
            Specify whether to make scales of both axis equal. Set to False
            by default. The default is False.

        Raises
        ------
        None

        Return
        ------
        None

        Notes
        -----
        To show the final ray diagram include matplotlib.pyploy.show() after
        this show() method.

        The plot has a default title and axis labels but they can be modified
        by redefining them after this show() method.
        """

        p_x = self._p[:, 2]
        p_y = self._p[:, 0]

        plt.title("Ray Diagram")
        plt.xlabel("z")
        plt.ylabel('x')

        if equal == True:
            plt.axis('equal')

        # plot all vertices of a ray if it doesnt miss an optical element
        if self._miss == False:
            plt.plot(p_x, p_y, marker='.', markevery=[0, -1], \
            color=self._color, linewidth=0.9)

        # plot vertices of a ray which misses an element up to that element
        if self._miss == True and miss==True:
            plt.plot(p_x, p_y, marker='x', markevery=[0, -1], \
            color=self._color, linewidth=0.9, alpha=0.4)

        return None


class Beam():
    """
    The Ray class to represent a beam of light as a collection of rays.

    Attributes
    ----------
    _rays : list
        List of all of the Ray class instances within the beam.

    _p : numpy.ndarray
        2D array represeing the initial position vector of each of the rays
        within the beam along axis 0.

    _k : numpy.ndarray
        1D array representing the initial direction of propagation vector of
        beam and therefore of all of the rays within the beam.

    _n : int
        Number of all of the rays within the beam.

    _prof : string
        String representing the type of profile of the beam.

    _color : string
        Color with which to plot the ray on the ray diagram.

    Methods
    -------
    __init__(self, p, k, r, n, profile, color='red')
        Make an instance of the Beam class.

    __repr__(self)
        Representation method for the Beam class.

    __str__(self)
        String method for the Beam class.

    reset(self)
        Reset the beam by moving to the initial position vector and direction
        of propagation.

    profile(self)
        Plot a 2D profile of the beam at the point where it is initialised.

    show(self, miss=True, equal=False)
        Plot the beam on a 2D ray diagram using the specified color.

    get_RMS(self)
        Get the RMS spot radius using the latest position vectors of the beam.

    get_fnum(self, dz, z_max=np.inf, show=False, color='blue')
        Find the position vector of the numerical focal point at which the RMS
        spot size of the beam is a local minimum.
    """

    def __init__(self, p, k, r, n, profile='circle', color='red'):
        """
        Make an instance of the Beam class.

        Parameters
        ----------
        p : numpy.ndarray or list
            Initial position vector of the beam.

        k : numpy.ndarray or list
            Initial direction of propagation vector of the beam.

        r : float
            Radius or length characterising the beam profile depending on
            which profile is used. Must be +ve.

        n : int
            Number of rays per the specified radius or length. Must be a
            +ve integer.

        profile : string
            Specify the beam profile. Can be 'circle', 'square', 'hline' with
            the default being 'circle'. See help for get_circ, get_sqr and
            get_hline for more information.

        color : string, optional
            Color with which to plot the beam. The default is 'red'.

        Raises
        ------
        ValueError
            When the input parameters r and n have inappropriate values.
            When the direction of propagation vector has a zero magnitude.

        ShapeError
            When the p and k used to initialise the beam have incorrect shapes.

        TypeError
            When the beam profile is specified to a nonexistent type.

        Returns
        -------
        None
        """

        p = np.array(p)
        k = np.array(k)

        if p.shape != (3,):
            raise ShapeError("Vector p has to be a 1D array of length 3.")

        if k.shape != (3,):
            raise ShapeError("Vector k has to be a 1D array of length 3.")

        if get_mag(k) == 0:
            raise ValueError("Direction of propagation vector must have a "
                             "non-zero magnitude.")

        if profile == 'circle':
            x, y = get_circ(r, n)
        elif profile == 'square':
            x, y = get_sqr(r, n)
        elif profile == 'hline':
            x, y = get_hline(r, n)
        else:
            raise TypeError("Profile has to be either 'circle', 'square' or " \
                            "'hline'.")

        n_rays = len(x)

        # shift x, y points from the distribution to the position of the beam
        p_beam = np.array([x + p[0], y + p[1], np.zeros(n_rays) + p[2]])

        # initialise rays within the beam and store as a list
        rays = []
        for i in range(0, n_rays):
            rays.append(Ray(p_beam[:, i], k, color))

        self._rays = rays
        self._p = p_beam
        self._k = k
        self._n = n_rays
        self._prof = profile
        self._color = color

        return None


    def __repr__(self):
        """
        Representation method for the Beam class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which represents all of attributes of the Ray class
            listed in the order _p, _k, _n, _prof, _color.
        """

        return "%s, %s, %s, %s, %s" \
            % (self._p, self._k, self._n, self._prof, self._color)


    def __str__(self):
        """
        String method for the Beam class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which shows all of attributes of the Beam class.
        """

        return "p = %s \n k = %s \n n = %s \n profile = %s \n color = %s" \
            % (self._p, self._k, self._n, self._prof, self._color)


    def reset(self):
        """
        Reset the beam by moving to the initial position vector and direction
        of propagation. Done by resetting each ray within the beam.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Return
        ------
        None
        """

        n = self._n

        for i in range(0, n):
            self._rays[i].reset()

        return None


    def profile(self):
        """
        Plot a 2D profile of the beam at the point where it is initialised.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        None
        """
        x = self._p[0, :]
        y = self._p[1, :]

        plt.title("Beam Profile")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')

        plt.plot(x, y, '.', color=self._color)

        return None


    def show(self, miss=True, equal=False):
        """
        Plot the beam on a 2D ray diagram using the specified color.

        Parameters
        ----------
        miss : bool, optional
            Specify whether to show rays which do not intercept all of the
            optical elements. The default is True.

        equal : bool, optional
            Specify whether to make scales of both axis equal. Set to False
            by default. The default is False.

        Raises
        ------
        None

        Return
        ------
        None

        Notes
        -----
        To show the final ray diagram include matplotlib.pyploy.show() after
        this show() method.

        The plot has a default title and axis labels but they can be modified
        by redefining them after this show() method.
        """

        for i in range(0, self._n):
            self._rays[i].show(miss, equal)

        return None


    def get_RMS(self):
        """
        Get the RMS spot radius using the latest position vectors of the rays
        within the beam.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        RMS : numpy.float64
            RMS spot radius of the beam.

        Notes
        -----
        The method will allways return the RMS spot radius at the surface of the
        optical element which the beam intercepts or is propagated through
        before the get_RMS() method is called. The surface can for example be a
        lens or a screen.
        """

        n_rays = self._n

        miss = np.empty(n_rays)
        p = np.empty((n_rays, 3))

        # extract the last position vectors and miss bools from all of the rays
        for i in range(0, n_rays):
            miss[i] = self._rays[i]._miss
            p[i, :] = self._rays[i]._p[-1, :]

        # only take position vectors of the rays which do not miss an element
        p = p[miss == False]
        n_p = len(p)

        p_mean = np.mean(p, axis=0)
        RMS = np.sqrt(np.sum(np.square(p - p_mean)) / n_p)

        return RMS


    def _get_RMS_new(p, k, z):
        """
        Private method which returns the RMS value using a plane at a given
        position along the z axis. Used in get_fnum().
        """

        l = np.abs((z - p)[:,2] / k[:,2])

        r = p + k * l[:, None]

        r_mean = np.mean(r, axis=0)
        n_r = len(r)

        RMS = np.sqrt(np.sum(np.square(r - r_mean)) / n_r)

        return RMS


    def get_fnum(self, dz, z_max=np.inf, show=False, color='blue'):
        """
        Find the position vector of the numerical focal point at which the RMS
        spot size of the beam is a local minimum.

        Parameters
        ----------
        dz : float
            Value by which to increment the position along the z axis when
            finding the local minimum of the RMS spot radius.

        z_max : float, optional
            Maximum position along the z axis at which the method stops trying
            to find the focal point. The default is np.inf so no limit.

        show : bool, optional
            Specify whether to show the numerical focal point on the ray
            diagram. The default is False.

        color : bool, optional
            Specify the color with which to show the numerical focal point.
            The default is 'blue'.

        Raises
        ------
        PhysicalError
            When rays diverge in the direction of dz or value of dz is too
            large so RMS spot radius is increasing.

        Returns
        -------
        r_focus : numpy.ndarray
            Position vector of the numerical focal point.

        RMS_focus : numpy.float64
            Minimum value of the RMS spot radius at the focal point.

        Notes
        -----
        The expected uncertanty in the z coordinate of the numerical focal
        point is given by ± dz.

        The method will allways start finding the numerical focal point from
        the surface of the optical element which the beam is propagated through
        before the get_focus() method is called. The method can not be called
        after optical elements which do not propagate the beam such as a screen.

        This method can only find numerical focal points for beams which
        converge along the z axis.
        """

        n_rays = self._n

        miss = np.empty(n_rays)
        p = np.empty((n_rays, 3))
        k = np.empty((n_rays, 3))

        # extract attributes of rays just like in get_RMS()
        for i in range(0, n_rays):
            miss[i] = self._rays[i]._miss
            p[i, :] = self._rays[i]._p[-1, :]
            k[i, :] = self._rays[i]._k[-1, :]

        p = p[miss == False]
        k = k[miss == False]

        # position vector at of plane at which to find the RMS spot radius
        z = np.array([0, 0, np.amax(p[:, 2])])
        RMS = Beam._get_RMS_new(p, k, z)

        z[2] += dz
        RMS_new = Beam._get_RMS_new(p, k, z)

        if RMS_new >= RMS:
            raise PhysicalError("Rays diverge in the direction of dz or dz " \
                                "is too large.")

        # addvance the plane and calcualte RMS spot radius untill minimum found
        while RMS_new <= RMS and z[2] < z_max:
            RMS = RMS_new

            z[2] += dz
            RMS_new = Beam._get_RMS_new(p, k, z)

        RMS_focus = RMS

        # take position of focal point to be the average position of rays within
        # a beam at the approximate minimum
        z[2] -= dz/2
        l = np.abs((z - p)[:,2] / k[:,2])
        r = p + k * l[:, None]
        r_focus = np.mean(r, axis=0)

        if show == True:
            plt.axvline(x=r_focus[2], ls=':', color=color)
            plt.plot(r_focus[2], r_focus[0], "X", color=color, zorder=np.inf)

        return r_focus, RMS_focus


class PropElem():
    """
    The PropElem base class for all optical elements which propagate light.

    Attributes
    ----------
    None

    Methods
    -------
    get_p(self)
        Return the position vector used to define a propagating surface.

    get_n(self)
        Return the normal vector to the principal plane of a propagating surface
        which defines its orientation.

    intercept(self, light)
        Intercept ray or beam of light with the propagating surface.

    propagate(self, light)
        Propagate ray or beam of light through the propagating surface.

    Note
    ----
    Propagating surfaces are refractive or reflective surfaces which propagate
    light for example lenses or mirrors. For now the propagate() method refracts
    light through a surface.
    """

    def get_p(self):
        """
        Return the position vector used to define a propagating surface.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        p : numpy.ndarray
            Position vector of the propagating surface.
        """

        p = self._p

        if isinstance(self, SpehrSurf):
            p[2] - self._R

        return p


    def get_n(self):
        """
        Return the normal vector to the principal plane of a propagating surface
        which defines its orientation.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        n : numpy.ndarray
            Normal vector to the principal plane of the propagating surface.
        """

        if isinstance(self, FlatSurf):
            n = self._n
        else:
            n = np.array([0, 0, -1])

        return n


    def _get_t(R, apt):
        """
        Private method which returns the distance along z between the edge of
        an optical element and the center of the element. Used in SpherLens.
        """

        if np.isnan(R):
            t = 0
        else:
            t = R - np.sign(R) * np.sqrt(R**2 - apt**2)

        return t


    def _make_surf(p, R, n1, n2, apt, color='blue'):
        """
        Private method for initialising either a flat or a spherical
        propagating surface. Used in SpherLens.
        """

        if np.isnan(R):
            surf = FlatSurf(p, [0, 0, -1], n1, n2, apt, color)
        else:
            surf = SpherSurf(p, R, n1, n2, apt, color)

        return surf


    def intercept(self, light):
        """
        Intercept ray or beam of light with the propagating surface.

        Parameters
        ----------
        light : ORT.Ray or ORT.Beam
            Ray or beam which is to be intercepted with the propagating
            surface. Has to be either ORT.Ray or ORT.Beam.

        Raises
        ------
        PhysicalError
            Light is propagating away from the optical element.

        TypeError
            When light is not an instance of ORT.Ray or ORT.Beam.

        Returns
        -------
        l : numpy.float64 or numpy.ndarray
            Length of the vector from the current position vector of the ray to
            the point of interception with the propagating surface. If light is
            an instance of ORT.Beam then l is an array of lengths for each ray
            within the beam.
        """

        if isinstance(light, Ray):
            l = self._intercept_ray(light)
        elif isinstance(light, Beam):
            l = np.empty(light._n)

            for i in range(0, light._n):
                l[i] = self._intercept_ray(light._rays[i])
        else:
            raise TypeError("light has to be either ORT.Ray or ORT.Beam.")

        return l


    def propagate(self, light):
        """
        Propagate ray or beam of light through the propagating surface.

        Parameters
        ----------
        light : ORT.Ray or ORT.Beam
            Ray or beam which is to be intercepted with the propagating
            surface. Has to be either ORT.Ray or ORT.Beam.


        Raises
        ------
        PhysicalError
            When incident ray undergoes total internal reflection.

        TypeError
            When light is not an instance of ORT.Ray or ORT.Beam.

        Returns
        -------
        None
        """

        if isinstance(light, Ray):
            self._propagate_ray(light)
        elif isinstance(light, Beam):
            for i in range(0, light._n):
                self._propagate_ray(light._rays[i])
        else:
            raise TypeError("light has to be either ORT.Ray or ORT.Beam.")

        return None


class FlatSurf(PropElem):
    """
    The FlatSurf subclass to represent a flat refracting surface.

    Attributes
    ----------
    _p : numpy.ndarray
        1D array representing the position vector of the center of the
        flat surface.

    _n : numpy.ndarray
        1D array representing the normal vector to the flat surface.

    _n1 : float
        Refractive index on the incident ray side of the surface.

    _n2 : float
        Refractive index on the transmitted ray side of the surface.

    _apt : float
        Aperture radius of the flat surface.

    _color : string
        Color with which to plot the flat surface on a ray diagram.

    Methods
    -------

    __init__(self, p, n1, n2, apt, color='blue')

    __repr__
        Representation method for the FlatSurf class.

    __str__
        String method for the FlatSurf class.

    show()
        Plot the flat surface on a 2D ray diagram using the specified color.
    """

    def __init__(self, p, n, n1, n2, apt, color='blue'):
        """
        Make an instance of the FlatSurf class.

        Parameters
        ----------
        p : numpy.ndarray or list
            Position vector of the center of the flat surface.

        n : numpy.ndarray or list
            Normal vector to the flat surface.

        n1 : float
            Refractive index on the incident ray side of the surface.

        n2 : float
            Refractive index on the transmitted ray side of the surface.

        apt : float
            Radius of apertue of the flat surface.

        color : string, optional
            Color with which to plot the flat surface on a ray diagram. The
            default is 'blue'.

        Raises
        ------
        ShapeError
            When the p used to initialise the flat surface has incorrect shape.

        ValueError
            When the normal vector n has a zero magnitude.

        Returns
        -------
        None

        Notes
        -----
        The normal is defined against the direction of propagation of the
        incoming ray. If the ray is propagating in the +ve z direction the
        normal will be defined in the -ve z direction.
        """

        p = np.array(p)
        n = np.array(n)

        if p.shape != (3,):
            raise ShapeError("Vector p has to be a 1D array of length 3.")

        if n.shape != (3,):
            raise ShapeError("Vector n has to be a 1D array of length 3.")

        if get_mag(n) == 0:
            raise ValueError("Normal to screen vecor must have a non-zero "
                             "magnitude.")

        n = get_norm(n)

        self._p = p
        self._n = n
        self._n1 = n1
        self._n2 = n2
        self._apt = apt
        self._color = color

        return None


    def __repr__(self):
        """
        Representation method for the FlatSurf class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which represents all of attributes of the FlatSurf class
            listed in the order _p, _n, _n1, _n2, _apt, _color.
        """

        return "%s, %s, %s, %s, %s, %s" \
            % (self._p, self._n, self._n1, self._n2, self._apt, self._color)


    def __str__(self):
        """
        String method for the FlatSurf class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which shows all of attributes of the FlatSurf class.
        """

        return "p = %s \n n = %s \n n1 = %s \n n2 = %s \n apt = %s \n color "\
               "= %s" \
               % (self._p, self._n, self._n1, self._n2, self._apt, self._color)


    def _intercept_ray(self, ray):
        """
        Protected method to intercept a ray with the flat surface.
        Used in intercept().
        """

        p_surf = self._p
        n_surf = self._n
        p_ray = ray._p[-1]
        k_ray = ray._k[-1]

        # calculate the lenth along the ray at which it intercepts the element
        l = np.dot(n_surf, (p_surf - p_ray)) / np.dot(n_surf, k_ray)

        return l


    def _propagate_ray(self, ray):
        """
        Protected method to propagate a ray through the flat surface.
        Used in propagate().
        """

        p_surf = self._p
        n_surf = self._n

        n1 = self._n1
        n2 = self._n2
        apt = self._apt

        p_ray = ray._p[-1]
        k_ray = ray._k[-1]

        l = self._intercept_ray(ray)

        p_new = p_ray + l * k_ray
        k_new = snell(k_ray, n_surf, n1, n2)

        # if a ray misses the element truncate it by setting the new direction
        # of travel to an arrya of nan
        if get_mag(p_new - p_surf) > apt:
            k_new = np.nan * np.empty(3)
            ray._miss = True

        if l < 0:
            raise PhysicalError("Light is propagating away from the optical " \
                                "element.")

        ray.append(p_new, k_new)

        return None


    def show(self):
        """
        Plot the flat surface on a 2D ray diagram using the specified color.

        Rauises
        -------
        PhysicalError
            When flat surface is parallel to the x-y plane so can not be imaged
            on a plot of x against y.

        Raises
        ------
        PhysicalError
            When the flat surface is parallel to the x-z plane so can not be
            shown.

        Returns
        -------
        None
        """

        p = self._p
        n = self._n
        apt = self._apt

        # calculate coordinates of the endpoints of the surface in a plane
        # parallel to the x-z plane
        if n[0] == 0:
            x = p[2] * np.array([1, 1])
            y = p[0] + apt * np.array([-1, 1])
        elif n[2] == 0:
            x = p[2] + apt * np.array([1, -1])
            y = p[0] * np.array([1, 1])
        elif n[0] != 0 and n[2] != 0:
            x = p[2] + apt * n[0] * np.array([1, -1])
            y = p[0] + apt * n[2] * np.array([-1, 1])
        else:
            raise PhysicalError("Flat surface is parallel to the x-z plane so" \
                                "can not be show.")

        plt.plot(x, y, color=self._color)

        return None


class SpherSurf(PropElem):
    """
    The SpherSurf subclass to represent a sherical propagating surface.

    Attributes
    ----------
    _p : numpy.ndarray
        1D array representing the position vector of the center of the sphere
        defining the spherical surface. Different to the p used to initialise.

    _R : float
        Radius of curvature of the spherical surface.

    _n1 : float
        Refractive index on the incident ray side of the surface.

    _n2 : float
        Refractive index on the transmitted ray side of the surface.

    _apt : float
        Aperture radius of the spherical surface.

    _color : string
        Color with which to plot the spherical surface on a ray diagram.

    Methods
    -------
    __init__(self, p, R, n1, n2, apt, color='blue')

    __repr__
        Representation method for the SpherSurf class.

    __str__
        String method for the SpherSurf class.

    show()
        Plot the spherical surface on a 2D ray diagram using the specified
        color.

    get_fparax(show=False)
        Calculate the paraxial focal length of the spherical surface.
    """

    def __init__(self, p, curv, n1, n2, apt, color='blue'):
        """
        Make an instance of the SpherSurf class.

        Parameters
        ----------
        p : numpy.ndarray or list
            Position vector of the optical center of the spherical surface.

        curv : float
            Curvature of the spherical surface. Must be non-zero.

        n1 : float
            Refractive index on the incident ray side of the surface.

        n2 : float
            Refractive index on the transmitted ray side of the surface.

        apt : float
            Radius of apertue of the flat surface.

        color : string, optional
            Color with which to plot the flat surface on a ray diagram. The
            default is 'blue'.

        Raises
        ------
        ValueError
            When the curvature used to initialise the spherical surface is 0.

        PhysicalError
            When the aperture used to initialise the spherical surface is
            greater than the radius of curvature of the surface.

        ShapeError
            When the position used to initialise the spherical surface has an
            incorrect shape.

        Returns
        -------
        None
        """

        if curv == 0:
            raise ValueError("Curvature of SpherSurf can not be 0. For a " \
                             "flat surface use the FlatSurf class.")

        R = get_R(curv)

        if apt > np.abs(R):
            raise PhysicalError("Aperture can not be greater than the " \
                                "radius the surface.")

        p = np.array(p)

        if p.shape != (3,):
            raise ShapeError("Vector p has to be a 1D array of length 3.")

        p[2] += R
        self._p = p

        self._R = R
        self._n1 = n1
        self._n2 = n2
        self._apt = apt
        self._color = color

        return None


    def __repr__(self):
        """
        Representation method for the SpherSurf class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which represents all of attributes of the SpherSurf class
            listed in the order _p, _R, _n1, _n2, _apt, _color.
        """

        return "%s, %s, %s, %s, %s, %s, %s" \
            % (self._p, self._R, self._n1, self._n2, self._apt, self._color)


    def __str__(self):
        """
        String method for the SpherSurf class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which shows all of the attributes of the SpherSurf class.
        """

        return "p = %s \n R = %s \n n1 = %s \n n2 = %s \n apt = %s \n" \
               "color = %s" \
               % (self._p, self._R, self._n1, self._n2, self._apt, self._color)


    def _intercept_ray(self, ray):
        """
        Protected method to intercept a ray with the spherical surface.
        Used in intercept().
        """

        p_surf = self._p
        R = self._R
        p_ray = ray._p[-1]
        k_ray = ray._k[-1]

        r = p_ray - p_surf

        to_sqrt = np.square(np.dot(r, k_ray)) - np.dot(r,r) + R**2

        # check if ray intercepts the sphere otherwise truncate by returning nan
        if to_sqrt >= 0:
            l = - np.dot(r, k_ray) - np.sign(R) * np.sqrt(to_sqrt)
        else:
            l = np.nan

        return l


    def _propagate_ray(self, ray):
        """
        Protected method to propagate a ray through the spherical surface.
        Used in propagate().
        """

        p_surf = self._p
        R = self._R
        n1 = self._n1
        n2 = self._n2
        apt = self._apt

        p_ray = ray._p[-1]
        k_ray = ray._k[-1]

        l = self._intercept_ray(ray)

        n = p_ray + l * k_ray - p_surf
        n = np.sign(R) * get_norm(n)

        p_new = p_ray + l * k_ray
        k_new = snell(k_ray, n, n1, n2)

        if l < 0:
            raise PhysicalError("Rays are propagating away from optical " \
                                "element.")

        # if ray misses the spherical surface set new position to be the
        # intercept with a plane in front of the spherical surface and truncate
        # the ray by setting new direction of travel to an array of nan
        if np.isnan(l) or get_mag((p_new - p_surf)[:2]) > apt:
            l = np.abs(((p_surf - p_ray)[2] - R) / k_ray[2])
            p_new = p_ray + l * k_ray
            k_new = np.nan * np.empty(3)
            ray._miss = True

        ray.append(p_new, k_new)

        return None


    def show(self):
        """
        Plot the spherical surface on a 2D ray diagram using the specified
        color.

        Raises
        -------
        None

        Returns
        -------
        None
        """

        p = self._p
        R = self._R
        apt = self._apt

        # use equation of circle to trace the spherical surface at a crossection
        # parallel to the x-z plane
        y_arr = np.linspace(-apt, apt, 100)
        x_arr = - np.sign(R) * np.sqrt(R**2 - y_arr**2)

        plt.plot(x_arr + p[2], y_arr + p[0], color=self._color)

        return None


    def get_fparax(self, show=False):
        """
        Calculate the paraxial focal length of the spherical surface.

        Parameters
        ----------
        show : bool, optional
            Specify whether to show the calcualted paraxial focal length on the
            ray diagram. The default is False.

        Raises
        -------
        None

        Returns
        -------
        f : float
            Calculated paraxial focal length.
        """

        p = self._p
        R = self._R
        n1 = self._n1
        n2 = self._n2

        f = R * n2 / (n2 - n1)

        if show == True:
            plt.axvline(x=p[2] - R + f, ls=':', color=self._color)

        return f


class SpherLens(PropElem):
    """
    The SpherLens subclass to represent a lens with spherical surfaces.

    Attributes
    ----------
    _p : numpy.ndarray
        1D array representing the position vector of the center of the lens.

    _z : float
        Thickness of the lens defined along the optical center of the lens.

    _curv1 : float
        Curvature of the first spherical surface (on the left side).

    _curv2 : float
        Curvature of the second spherical surface (on the right side).

    _apt : float
        Aperture radius of the spherical lens.

    _color : string
        Color with which to plot the spherical lens on a ray diagram.

    _surf1 : ORT.SpherSurf or ORT.FlatSurf
        Either a spherical surface or a flat surface which represents the first
        surface making up the lens (on the right side).

    _surf2 : ORT.SpherSurf or ORT.FlatSurf
        Either a spherical surface or a flat surface which represents the
        second surface making up the lens (on the right side).

    _t1 : float
        Distance along the z axis of the edge of the first surface from the
        center of the lens.

    _t2 : float
        Distance along the z axis of the edge of the first surface from the
        center of the lens.

    Methods
    -------
    __init__(self, p, z, curv1, curv2, n1, n2, apt, color='blue')
        Make an instance of the SpherLens class.

    __repr__(self)
        Representation method for the SpherLens class.

    __str__(self)
        String method for the SpherLens class.

    show(self)
        Plot the spherical lens on a 2D ray diagram using the specified color.

    get_fparax(self, show=False)
        Calculate the paraxial focal length of the spherical lens.

    get_shape(self)
        Calculate the Coddington shape factor for the spherical lens.

    Notes
    -----
    _t1 and _t2 are used for checking if the surfaces intercept and for
    plotting of the spherical lens.
    """

    def __init__(self, p, z, curv1, curv2, n1, n2, apt, color='blue'):
        """
        Make an instance of the SpherLens class.

        Parameters
        ----------
        p : numpy.ndarray or list
            Position vector of the center of the lens.

        z : float
            Thickness of the lens defined along the optical center of the lens.

        curv1 : float
            Curvature of the first spherical surface (on the left side).

        curv2 : float
            Curvature of the second spherical surface (on the right side).

        n1 : float
            Refractive index of the medium surrounding the spherical lens.

        n2 : float
            Refractive index of the spherical lens.

        apt : float
            Aperture radius of the spherical lens.

        color : string, optional
            Color with which to plot the spherical lens on a ray diagram. The
            default is 'blue'.

        Raises
        ------
        PhysicalError
            When the aperture used to initialise the spherical lens is greater
            than the smaller radius of curvature of the spherical lens.

        PhysicalError
            Wheen the two surfaces of the spherical lens intersect.

        ShapeError
            When the position used to initialise the spherical lens has an
            incorrect shape.

        Returns
        -------
        None
        """

        R1 = get_R(curv1)
        R2 = -get_R(curv2)

        if apt > np.abs(R1) or apt > np.abs(R2):
            raise PhysicalError("Aperture radius can not be greater than the " \
                                "smaller radius of curvature of the " \
                                "spherical lens.")

        t1 = PropElem._get_t(R1, apt)
        t2 = PropElem._get_t(R2, apt)

        if z < np.abs(t1 - t2):
            raise PhysicalError("The two surfaces of the spherical lens " \
                                "intersect.")

        p = np.array(p)

        if p.shape != (3,):
            raise ShapeError("Vector p has to be a 1D array of length 3.")

        p1 = p - np.array([0, 0, z/2])
        p2 = p + np.array([0, 0, z/2])

        self._surf1 = PropElem._make_surf(p1, 1/R1, n1, n2, apt, color)
        self._surf2 = PropElem._make_surf(p2, 1/R2, n2, n1, apt, color)

        self._p  = p
        self._z = z
        self._curv1 = curv1
        self._curv2 = curv2
        self._apt = apt
        self._t1 = t1
        self._t2 = t2
        self._color = color

        return None


    def __repr__(self):
        """
        Representation method for the SpherLens class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which represents all of attributes of the SpherLens class,
            except _surf1 and _surf2, listed in the order _p, _z, _curv1, curv2,
            _apt, _color, _t1, _t2.
        """

        return "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s" \
               % (self._p, self._z, self._curv1, self._curv2, self._apt, \
               self._color, self._t1, self._t2)


    def __str__(self):
        """
        String method for the SpherLens class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which shows all of the attributes of the SpherLens class
            except _surf1 and _surf2.
        """

        return "p = %s \n z = %s \n curv_1 = %s \n curv2 = %s \n apt = %s \n" \
               "color = %s \n t1 = %s \n t2 = %s" \
               % (self._p, self._z, self._curv1, self._curv2, self._apt, \
               self._color, self._t1, self._t2)


    def _intercept_ray(self, light):
        """
        Protected method to intercept a ray with the first surface of the
        spherical lens. Used in intercept().
        """
        l = self._surf1._intercept_ray(light)

        return l


    def _propagate_ray(self, light):
        """
        Protected method to propagate a ray through the spherical surface.
        Used in propagate().
        """
        self._surf1._propagate_ray(light)
        self._surf2._propagate_ray(light)

        return None


    def show(self):
        """
        Plot the whole spherical lens on a 2D ray diagram using the specified
        color.

        Rauises
        -------
        None

        Returns
        -------
        None
        """

        p = self._p
        z = self._z
        apt = self._apt
        t1 = self._t1
        t2 = self._t2

        # calculate coordinates of the endpoints of lines which join the two
        # suefaces of the lens
        y_top = p[0] + apt * np.ones(2)
        y_bot = p[0] - apt * np.ones(2)
        x = p[2] + np.array([- z/2 + t1, z/2 + t2])

        # show the two surfaces and join them with lines
        self._surf1.show()
        self._surf2.show()
        plt.plot(x, y_top, color=self._color)
        plt.plot(x, y_bot, color=self._color)

        return None


    def get_fparax(self, show=False):
        """
        Calculate the paraxial focal length of the spherical lens.

        Parameters
        ----------
        show : bool, optional
            Specify whether to show the calcualted paraxial focal length on the
            ray diagram. The default is False.

        Raises
        ------
        None

        Returns
        -------
        f : float
            Calculated paraxial focal length.
        """

        p = self._p
        z = self._z
        curv1 = self._curv1
        curv2 = -self._curv2
        n1 = self._surf1._n1
        n2 = self._surf1._n2

        # calculate the paraxial focus of a thick lens
        f = curv1 - curv2 + z * (n2 / n1 - 1) / n2 * curv1 * curv2
        f = 1 / f / (n2 / n1 - 1)

        if show == True:
            plt.axvline(x = p[2] + f, ls=':', color=self._color)

        return f


    def get_shape(self):
        """
        Calculate the Coddington shape factor for the spherical lens.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        q : float
            Calculated Coddington shape factor of the lens.
        """

        curv1 = self._curv1
        curv2 = self._curv2

        if curv1 == 0 and curv2 ==0:
            q = 0
        else:
            q = (curv1 - curv2) / (curv1 + curv2)

        return q


class Screen():
    """
    The Screen class to represent a screen on which light is imaged.

    Attributes
    ----------
    _p : numpy.ndarray
        1D array representing the position vector of the screen.

    _n : numpy.ndarray
        1D array representing the normal vector to the plane of the screen.

    _color : string
        Color with which to plot the screen on the ray diagram.

    _virt : bool
        Specify whether to intercept and image rays which form virtual images.

    Methods
    -------
    __init__(self, p, n, color='black')
        Make an instance of the Beam class.

    __repr__(self)
        Representation method for the Screen class.

    __str__(self)
        String method for the Screen class.

    intercept(self, light)
        Intercept a ray or a beam of light with the screen.

    image(self, light)
        Plot the image formed by a ray or beam of light at the screen.

    show(self)
        Plot the screen on a 2D ray diagram using the specified color.

    """

    def __init__(self, p, n, virtual=False, color='black'):
        """
        Make an instance of the Beam class.

        Parameters
        ----------
        p : numpy.ndarray or list
            Position vector of the screen.

        n : numpy.ndarray or list
            Normal vector to the plane of the screen.

        color : string, optional
            Specify the color with which to plot the screen on the ray diagram.
            The default is 'black'.

        virt : bool, optional
            Specify whether to intercept and image rays which form virtual
            images. The default is False.

        Raises
        ------
        ShapeError
            When the p and k used to initialise the screen have incorrect
            shapes.

        ValueError
            When the normal vector n has a zero magnitude.

        Returns
        -------
        None

        Notes
        -----
        The normal is defined against the direction of propagation of the
        incoming ray. If the ray is propagating in the +ve z direction the
        normal will be defined in the -ve z direction.
        """

        p = np.array(p)
        n = np.array(n)

        if p.shape != (3,):
            raise ShapeError("Vector p has to be a 1D array of length 3.")

        if n.shape != (3,):
            raise ShapeError("Vector n has to be a 1D array of length 3.")

        if get_mag(n) == 0:
            raise ValueError("Normal to screen vecor must have a non-zero "
                             "magnitude.")

        n = get_norm(n)

        self._p = p
        self._n = n
        self._color = color
        self._virt = virtual

        return None


    def __repr__(self):
        """
        Representation method for the Screen class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which represents all of attributes of the Screen class
            listed in the order _p, _n, _color, _virt.
        """

        return "%s, %s, %s" % (self._p, self._n, self._color, self._virt)


    def __str__(self):
        """
        String method for the Screen class.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        str : string
            String which shows all of attributes of the Screen class.
        """

        return "p = %s \n n = %s \n color = %s \n virtual = %s" \
            % (self._p, self._n, self._color, self._virt)


    def _intercept_ray(self, ray):
        """
        Protected method to intercept a ray with the screen. Used in
        intercept().
        """

        p_scrn = self._p
        n_scrn = self._n

        p_ray = ray._p[-1]
        k_ray = ray._k[-1]

        # calculate the lenght along a ray at which it intercepts the screen
        l = np.dot(n_scrn, (p_scrn - p_ray)) / np.dot(n_scrn, k_ray)

        if l < 0 and self._virt == False:
            raise PhysicalError("Formed image is virtual set virtual=True to " \
                                "intercept virtual rays.")

        # truncate the ray by updating direction of travel to array of nan
        p_new = p_ray + l * k_ray
        k_new = np.array([np.nan, np.nan, np.nan])

        ray.append(p_new, k_new)

        return None


    def intercept(self, light):
        """
        Intercept a ray or a beam of light with the screen.

        Parameters
        ----------
        light : ORT.Ray or ORT.Beam
            Ray or beam which is to be intercepted with the screen. Has to be
            either ORT.Ray or ORT.Beam.

        Raises
        ------
        TypeError
            When light is not an instance of ORT.Ray or ORT.Beam.

        PhysicalError
            When light propagates away from screen so image will be virtual.

        Returns
        -------
        None
        """

        if isinstance(light, Ray):
            self._intercept_ray(light)
        elif isinstance(light, Beam):
            for i in range(0, light._n):
                self._intercept_ray(light._rays[i])
        else:
            raise TypeError("light has to be either ORT.Ray or ORT.Beam.")

        return None


    def _image_ray(self, ray):
        """
        Protected method to image a ray with the screen. Used in image().
        """

        x = ray._p[-1, 0]
        y = ray._p[-1, 1]

        plt.plot(x, y, '.', color=ray._color)

        return None


    def image(self, light):
        """
        Plot the image formed by a ray or beam of light at the screen.

        Parameters
        ----------
        light : ORT.Ray or ORT.Beam
            Ray or beam which forms the image on the screen. Has to be either
            ORT.Ray or ORT.Beam.

        Raises
        ------
        TypeError
            When light is not ORT.Ray or ORT.Beam.

        Returns
        -------
        None
        """

        plt.title("Image")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')

        if isinstance(light, Ray):
            self._image_ray(light)
        elif isinstance(light, Beam):
            for i in range(0, light._n):
                self._image_ray(light._rays[i])
        else:
            raise TypeError("light has to be either ORT.Ray or ORT.Beam.")

        return None


    def show(self):
        """
        Plot the screen on a 2D ray diagram using the specified color.

        Parameters
        ----------
        None

        Raises
        ------
        PhysicalError
            When the screen is parallel to the x-z plane so can not be shown.

        Returns
        -------
        None
        """

        p = self._p
        n = self._n

        if n[0] == 0 and n[2] == 0:
            raise PhysicalError("Screen is parallel to the x-z plane so can" \
                                "not be show.")

        # plot the screen as a line at which it intersects the x-z plane
        if n[2] == 0:
            plt.axvline(x=p[2], color=self._color)
        else:
            plt.axline((p[2], p[0]), slope=-n[2]/n[0], color=self._color)

        return None
