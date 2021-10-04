"""
Cloud-agnostic wrapper for object stores (GCS, S3, Azure Blob)
TODO: 
- Support AWS and Azure. Consider rewriting to tensorflows GFile API
  https://www.tensorflow.org/api_docs/python/tf/io/gfile/GFile
- Bug in gs-wrap: only works when copying a folder at least two levels deep in the bucket, otherwise dots in filenames are stripped
- Bug in gs-wrap: never create explicitly empty folders, these are reprsented by dummy objects and these cause problems
"""
import os
import sys
import tempfile
import urllib.parse
import shutil
import logging
import gswrap
import argparse
import urllib.request

S3_PREFIX = "s3://"
GS_PREFIX = "gs://"
AZURE_PREFIX = "az://"
HTTP_PREFIX = "http://"
HTTPS_PREFIX = "https://"

def is_uri(string):
    parsed_uri = urllib.parse.urlparse(string)
    return len(parsed_uri.scheme) > 0


def cp(src, dst):
    if src.startswith(HTTP_PREFIX) or src.startswith(HTTPS_PREFIX):
        if not is_uri(dst):
            # Simple http download
            if os.path.isdir(dst):
                dst = os.path.join(dst, _file_or_folder_name(src))
            _http_download(src, dst)
        else:
            with mount(src) as tempsrc:
                cp(tempsrc, dst)
    elif src.startswith(GS_PREFIX) or dst.startswith(GS_PREFIX):
        # Copy from or to GCS
        _gcs_copy(src, dst)
    elif src.startswith(S3_PREFIX) or dst.startswith(S3_PREFIX):
        # Copy from or to AWS S3
        raise NotImplementedError("S3 storage not yet implemented")
    elif not is_uri(src) and not is_uri(src):
        # Simple file copy
        _local_copy(src, dst)
    else:
        raise NotImplementedError(f"Unsupported copy operation between {src} and {dst}")


class mount:
    def __init__(self, path_or_uri, mode='r', is_dir=False):
        self.path_or_uri = path_or_uri
        self.mode = mode
        self.is_uri = is_uri(path_or_uri)
        self.is_dir = is_dir
        self.temp_dir = None
        if self.is_uri:
            self.temp_dir = tempfile.mkdtemp()
            self.local_path = os.path.join(self.temp_dir, _file_or_folder_name(path_or_uri))
            if self.mode == 'r':
                cp(path_or_uri, self.temp_dir)
            elif self.mode == 'w':
                if self.path_or_uri.startswith(HTTP_PREFIX) or self.path_or_uri.startswith(HTTPS_PREFIX):
                    raise Exception("Cannot write to HTTP(S)")
                if self.is_dir:
                    # Use expects a directory to write to, create it
                    os.mkdir(self.local_path)

        else:
            self.local_path = path_or_uri

    def __enter__(self):
        return self.local_path

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.is_uri:
            if self.mode == 'w' and exc_type is None:
                cp(self.local_path, self.path_or_uri)
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                logging.debug("Removing tempdir %s failed", self.local_path)
                pass
        if exc_type:
            return False
        return True


def _file_or_folder_name(path_or_uri):
    first, last = os.path.split(path_or_uri)
    if last == '':
        first, last = os.path.split(first)
    return last


def _gcs_copy(src, dst):
    client = gswrap.Client()
    client.cp(src, dst, recursive=True, multithreaded=True)


def _local_copy(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy(src, dst)


def _http_download(src, dst):
    assert src.startswith(HTTP_PREFIX) or src.startswith(HTTPS_PREFIX)
    assert not is_uri(dst)
    urllib.request.urlretrieve(src, dst)


def _test():
    # File mount and upload example
    with mount("gs://bucket/folder1/folder2/file.yaml") as source:
        with mount('gs://bucket/newfile.yaml', 'w') as target:
            shutil.copyfile(source, target)

    # Folder read and write mount example
    with mount("gs://bucket/folder1/folder2/") as source:
        with mount('gs://bucket/folder3/folder4', 'w', is_dir=True) as target:
            shutil.copy(source + '/file.yaml', target)


if __name__ == "__main__":

    example_text = """Examples:

    python -m bucketfs cp gs://my-project/mydataset .
    """

    parser = argparse.ArgumentParser(prog='python -m bucketfs',
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(help='Commands')
    parser_cp = subparsers.add_parser('cp', help='Copy files/folder from/to any cloud bucket')
    parser_cp.add_argument('src', type=str, help='Source (local path or s3://, gs://)')
    parser_cp.add_argument('dst', type=str, help='Destination (local path or s3://, gs://)')
    parser_cp.set_defaults(func=lambda args: cp(args.src, args.dst))

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)
