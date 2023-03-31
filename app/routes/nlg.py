import logging
from flask import Blueprint, jsonify, request
from app.services import DescriptionGeneration
from app.services import DescriptionGenerationNew
from app.core import limiter
from flask_limiter.util import get_remote_address
import flask_limiter

text_generation = Blueprint('text_generation', __name__)

logger = logging.getLogger(__name__)

@text_generation.route('/generate', methods=['POST'])
def get_text():
    request_data = request.get_json()
    data = DescriptionGeneration.get_generated_text(request_data)
    return jsonify(data)

@text_generation.route('/generate_new', methods=['POST'])
@limiter.limit('5/day',key_func = flask_limiter.util.get_ipaddr)
def get_text_new():
    request_data = request.get_json()
    data = DescriptionGenerationNew.get_generated_text(request_data)
    if not data:
        data = {}
    return jsonify(data)

