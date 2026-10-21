{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. add toctree option to make autodoc generate the pages

.. autoclass:: {{ objname }}

{% block attributes %}
{% if attributes | reject('in', inherited_members) | list %}
Attributes table
~~~~~~~~~~~~~~~~

.. autosummary::
{% for item in attributes if item not in inherited_members %}
    ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods | reject('in', inherited_members) | list %}
Methods table
~~~~~~~~~~~~~

.. autosummary::
{% for item in methods if item not in inherited_members %}
    {%- if item != '__init__' %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_documentation %}
{% if attributes | reject('in', inherited_members) | list %}
Attributes
~~~~~~~~~~

{% for item in attributes if item not in inherited_members %}

.. autoattribute:: {{ [objname, item] | join(".") }}
{%- endfor %}

{% endif %}
{% endblock %}

{% block methods_documentation %}
{% if methods | reject('in', inherited_members) | list %}
Methods
~~~~~~~

{% for item in methods if item not in inherited_members %}
{%- if item != '__init__' %}

.. automethod:: {{ [objname, item] | join(".") }}
{%- endif -%}
{%- endfor %}

{% endif %}
{% endblock %}
