"""
__________.__                                      _______               
\______   \  |   ____  ______ __________   _____   \      \ _____ ___  __
 |    |  _/  |  /  _ \/  ___//  ___/  _ \ /     \  /   |   \\__  \\  \/ /
 |    |   \  |_(  <_> )___ \ \___ (  <_> )  Y Y  \/    |    \/ __ \\   / 
 |______  /____/\____/____  >____  >____/|__|_|  /\____|__  (____  /\_/  
        \/                \/     \/            \/         \/     \/      

Copyright (c) 2024 Interactions Lab
License: MIT
Authors: Anthony Song and Nathan Dennler, Cornell University & University of Southern California
Project Page: https://github.com/interaction-lab/BlossomNav.git

This is an exception for if the IP address and the port are not valid

"""

import re

class InvalidAddressError(Exception):
    """Exception raised for invalid IP address or port."""
    
    def __init__(self, address_type, address_value, message="Invalid address"):
        self.address_type = address_type
        self.address_value = address_value
        self.message = f"{message}: {address_type} is {address_value} and has unsupported type: {type(address_value)}"
        super().__init__(self.message)

    def validate_ip(ip):
        """Validate the IP address."""
        if type(ip) != str:
            raise InvalidAddressError("IP address", ip)
        ip_regex = re.compile(
            r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        )
        if not ip_regex.match(ip):
            raise InvalidAddressError("IP address", ip)

    def validate_port(port):
        """Validate the port number."""
        if type(port) != int or not (0 <= port <= 65535):
            raise InvalidAddressError("Port", port)