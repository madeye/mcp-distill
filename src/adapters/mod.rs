pub mod claude;
pub mod codex;

use serde_json::Value;

use crate::schema::{Provider, Turn};

/// Convert a provider-native request message into a canonical [`Turn`].
pub fn to_turn(provider: &Provider, message: &Value) -> Turn {
    match provider {
        Provider::Claude | Provider::Anthropic => claude::message_to_turn(message),
        Provider::Codex | Provider::Openai => codex::message_to_turn(message),
        Provider::Other => claude::message_to_turn(message),
    }
}

/// Convert a provider-native API *response* into a canonical assistant [`Turn`].
pub fn response_to_turn(provider: &Provider, response: &Value) -> Turn {
    match provider {
        Provider::Claude | Provider::Anthropic => claude::response_to_turn(response),
        Provider::Codex | Provider::Openai => codex::response_to_turn(response),
        Provider::Other => claude::response_to_turn(response),
    }
}
