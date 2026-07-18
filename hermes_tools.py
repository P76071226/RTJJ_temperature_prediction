import sys
import os
# Add the hermes-agent package to the path
hermes_agent_path = '/data/data/com.termux/files/home/hermes-agent'
if hermes_agent_path not in sys.path:
    sys.path.insert(0, hermes_agent_path)

def send_message(**kwargs):
    try:
        from hermes_agent.tools.send_message_tool import send_message_tool
        args = {}
        args.update(kwargs)
        if 'action' not in args:
            args['action'] = 'send'
        return send_message_tool(args)
    except Exception as e:
        # Fallback: try direct import
        try:
            from tools.send_message_tool import send_message_tool
            args = {}
            args.update(kwargs)
            if 'action' not in args:
                args['action'] = 'send'
            return send_message_tool(args)
        except Exception as e2:
            return {"error": f"Failed to import send_message_tool: {e}; {e2}"}
