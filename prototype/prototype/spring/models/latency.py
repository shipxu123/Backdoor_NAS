import os
import re
import ssl
import certifi
import urllib3
import mimetypes
import json
import click
from six.moves.urllib.parse import urlencode


class HTTPClient:

    def __init__(self, server_host, pool_size=4, max_parallel_size=4, verify_ssl=True, ssl_ca_cert=None):
        self.server = server_host + "/api/v1/latency"
        self.user_name = get_user_name()

        # cert_reqs
        cert_reqs = ssl.CERT_REQUIRED if verify_ssl is True else ssl.CERT_NONE

        # ca_certs, if not set certificate file, use Mozilla's root certificates
        ca_certs = ssl_ca_cert if ssl_ca_cert is not None else certifi.where()

        # https pool manager
        self.pool_manager = urllib3.PoolManager(
            num_pools=pool_size,
            maxsize=max_parallel_size,
            cert_reqs=cert_reqs,
            ca_certs=ca_certs,
            cert_file=None,
            key_file=None
        )

    def call(self, hardware_name, backend_name, data_type, batch_size, onnx_file, graph_name="", force_test=False):
        if not os.path.exists(onnx_file):
            raise Exception("File {} not existed!".format(onnx_file))

        try:
            # query tuple params
            query_params = [
                ('hardware_name', sanitize_str(hardware_name)),
                ('backend_name', sanitize_str(backend_name)),
                ('data_type', sanitize_str(data_type)),
                ('batch_size', int(batch_size)),
                ('user_name', sanitize_str(self.user_name)),
                ('graph_name', sanitize_str(graph_name)),
                ('force', 'true' if force_test is True else 'false')
            ]
            url = self.server + '?' + urlencode(query_params)

            # post params
            post_params = []
            with open(onnx_file, 'rb') as f:
                filename = os.path.basename(f.name)
                filedata = f.read()
                mimetype = (mimetypes.guess_type(filename)[0] or 'application/octet-stream')
                post_params.append(tuple(['onnx_file', tuple([filename, filedata, mimetype])]))

            # headers
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'multipart/form-data',
                'User-Agent': 'GPDB-client'
            }
            # must del headers['Content-Type'], or the correct Content-Type
            # which generated by urllib3 will be overwritten.
            del headers['Content-Type']

            response = self.pool_manager.request(
                'POST', url,
                fields=post_params,
                encode_multipart=True,
                preload_content=True,
                timeout=None,
                headers=headers
            )

            ret_val = json.loads(response.data)
            return ret_val

        except Exception as e:
            print("HTTPClient Call Error: {}!".format(e))


def get_user_name():
    if 'USER' in os.environ:
        user_name = os.environ['USER']
    elif 'USERNAME' in os.environ:
        user_name = os.environ['USERNAME']
    else:
        user_name = "unknown"
    return user_name


def sanitize_str(ss):
    return re.sub(r'[^0-9a-zA-Z_\.\-]', '', str(ss))


class Latency(object):
    """An information object to pass data between CLI functions."""

    def __init__(self):  # Note: This object must have an empty constructor.
        """Create a new instance."""
        self.server = 'http://10.10.40.93:32770/gpdb/'

    def call(self, hardware_name, backend_name, data_type, batch_size, onnx_file,
             graph_name="", force_test=False, print_info=False):
        http_client = HTTPClient(self.server)
        ret_val = http_client.call(hardware_name, backend_name, data_type, batch_size, onnx_file,
                                   graph_name=graph_name, force_test=force_test)
        if print_info:
            if ret_val is not None:
                print(json.dumps(ret_val))
            else:
                print("Server Not Running!")

        return ret_val


# pass_info is a decorator for functions that pass 'Info' objects.
#: pylint: disable=invalid-name
pass_cfg = click.make_pass_decorator(Latency, ensure=True)
default_cfg = Latency()


@click.option("--server", "-s", type=str, help="spring.models.latency server url",
              default=default_cfg.server, show_default=True)
@click.option('--hardware', 'hardware_name', type=str,
              help='target hardware', required=True)
@click.option('--backend', 'backend_name', type=str,
              help='target backend', required=True)
@click.option('--data_type', 'data_type', type=str,
              help='data type, eg: int8', required=True)
@click.option('--batch_size', 'batch_size', type=int,
              help='batch size, eg: 8', required=True)
@click.option('--model_file', 'model_file',
              help='source model file', required=True)
@click.option('--graph_name', 'graph_name', type=str,
              help='graph name', required=False, default="")
@click.option('--force_test', 'force_test', is_flag=True,
              help='force test without querying database')
@click.command()
@pass_cfg
def ctl(cfg: Latency, server, hardware_name, backend_name, data_type,
        batch_size, model_file, graph_name, force_test):
    """Latency command line tools"""
    cfg.server = server
    cfg.call(hardware_name, backend_name, data_type, batch_size, model_file,
             graph_name=graph_name, force_test=force_test, print_info=True)


def main():
    ctl()