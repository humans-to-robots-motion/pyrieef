#!/usr/bin/env python

# Copyright (c) 2018, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                        Jim Mainprice on Sunday June 13 2018


# PyKDL can be installed using
# conda install -c jf pykdl

from PyKDL import *
print "Creating Robotic Chain"
chain=Chain()
joint0=Joint(Joint.RotZ) 
frame0=Frame(Vector(0.2,0.3,0))
segment0=Segment(joint0,frame0)
chain.addSegment(segment0) 
#Inertia zero (Don't want to mess with dynamics yet)
joint1=joint0 #Iqual joint
frame1=Frame(Vector(0.4,0,0))
segment1=Segment(joint1,frame1)
chain.addSegment(segment1)
joint2=joint1 #Iqual joint
frame2=Frame(Vector(0.1,0.1,0))
segment2=Segment(joint2,frame2)
chain.addSegment(segment2)
