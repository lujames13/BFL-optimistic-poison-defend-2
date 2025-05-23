# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower client."""


from .app import ClientLike as ClientLike
from .app import run_client as run_client
from .app import start_client as start_client
from .app import start_numpy_client as start_numpy_client
from .app import to_client as to_client
from .app import start_eth_client as start_eth_client
from .client import Client as Client
from .numpy_client import NumPyClient as NumPyClient
from .eth_client.eth_client import EthClient as EthClient

__all__ = [
    "Client",
    "ClientLike",
    "NumPyClient",
    "run_client",
    "start_client",
    "start_numpy_client",
    "to_client",
    "EthClient",
    "start_eth_client",
]