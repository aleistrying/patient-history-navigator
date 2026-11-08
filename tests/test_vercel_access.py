"""Offline access-boundary tests. No serving endpoint or remote SDK is called."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vercel'))

def load(name):
    spec = importlib.util.spec_from_file_location('phn_' + name, ROOT / 'vercel/api' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ask, warm, patients = [load(name) for name in ('ask', 'warm', 'patients')]

class AccessTests(unittest.TestCase):
    def invoke(self, module, payload=None, headers=None, path='/api/test'):
        instance = module.handler.__new__(module.handler)
        body = json.dumps(payload).encode() if payload is not None else b'{}'
        instance.headers = {'Content-Length': str(len(body)), **(headers or {})}
        instance.path = path
        instance.rfile = io.BytesIO(body)
        instance.wfile = io.BytesIO()
        response = {}
        instance.send_response = lambda code: response.update(status=code)
        instance.send_header = lambda key, value: response.setdefault('headers', {}).update({key: value})
        instance.end_headers = lambda: None
        with patch('urllib.request.urlopen', side_effect=AssertionError('Network must remain unused')):
            (instance.do_POST if module is ask else instance.do_GET)()
        response['body'] = json.loads(instance.wfile.getvalue())
        return response, instance

    def test_unconfigured_access_fails_closed_on_every_route(self):
        with patch.dict(os.environ, {}, clear=True):
            for module in (ask, warm, patients):
                with self.subTest(route=module.__name__):
                    response, _ = self.invoke(module, {'validate_only': True})
                    self.assertEqual(response['status'], 503)
                    self.assertEqual(response['headers']['Cache-Control'], 'no-store')
                    self.assertNotIn('patients', response['body'])

    def test_wrong_and_nonstring_passwords_rejected(self):
        with patch.dict(os.environ, {'PHN_PASSWORD': 'test-only-passphrase'}, clear=True):
            for password in ('wrong', None, {}, 123):
                response, _ = self.invoke(ask, {'password': password, 'validate_only': True})
                self.assertEqual(response['status'], 401)

    def test_unicode_password_validation_is_offline(self):
        with patch.dict(os.environ, {'PHN_PASSWORD': 'test-é-🔑'}, clear=True):
            response, _ = self.invoke(ask, {'password': 'test-é-🔑', 'validate_only': True})
            self.assertEqual(response['status'], 200)
            self.assertEqual(response['body'], {'ok': True})

    def test_body_limit_rejects_before_reading(self):
        with patch.dict(os.environ, {'PHN_PASSWORD': 'test'}, clear=True):
            for length in ('-1', str(ask.MAX_BODY_BYTES + 1)):
                response, instance = self.invoke(ask, {}, {'Content-Length': length})
                self.assertEqual(response['status'], 413)
                self.assertEqual(instance.rfile.tell(), 0)

    def test_nonobject_json_and_invalid_length_rejected(self):
        for payload, headers in (([], {}), ('text', {}), ({}, {'Content-Length': 'invalid'})):
            response, _ = self.invoke(ask, payload, headers)
            self.assertEqual(response['status'], 400)

    def test_warm_query_password_cannot_authorize(self):
        with patch.dict(os.environ, {'PHN_PASSWORD': 'test-only-passphrase'}, clear=True):
            response, _ = self.invoke(warm, path='/api/warm?password=test-only-passphrase')
            self.assertEqual(response['status'], 401)

    def test_roster_requires_header_and_is_not_cached(self):
        with patch.dict(os.environ, {'PHN_PASSWORD': 'test-only-passphrase'}, clear=True):
            response, _ = self.invoke(patients)
            self.assertEqual(response['status'], 401)
            self.assertNotIn('patients', response['body'])
            response, _ = self.invoke(patients, headers={'X-PHN-Password': 'test-only-passphrase'})
            self.assertEqual(response['status'], 200)
            self.assertGreater(len(response['body']['patients']), 0)
            self.assertEqual(response['headers']['Cache-Control'], 'no-store')

    def test_generated_frontend_uses_headers_and_handles_denial(self):
        source = (ROOT / 'vercel/index.html').read_text()
        self.assertNotIn('/api/warm?password=', source)
        self.assertIn("fetch('/api/patients', {headers: {'X-PHN-Password': sitePassword}})", source)
        self.assertIn("fetch('/api/warm', {headers: {'X-PHN-Password': sitePassword}})", source)
        self.assertIn('!res.ok', source)

if __name__ == '__main__':
    unittest.main()
