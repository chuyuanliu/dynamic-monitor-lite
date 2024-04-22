(function (Log, $) {
    let socket = null;
    let $socket_switch = null;
    let $notification_permission = null;
    let $notification_states = {};

    function socketSwitch() {
        if (socket != null) {
            if (socket.connected) {
                socket.disconnect();
            } else {
                socket.connect();
            }
        }
    }
    function notificationSwitch(notify) {
        let target = $notification_states[notify];
        if (target.hasClass("active")) {
            target.removeClass("active");
        }
        else {
            target.addClass("active");
        }
    }
    function notificationPermission() {
        if ($notification_permission.hasClass("active")) {
            $notification_permission.removeClass("active");
        }
        else {
            if (Notification.permission == "granted") {
                $notification_permission.addClass("active");
            }
            else {
                Notification.requestPermission().then(result => {
                    if (result == "granted") {
                        $notification_permission.addClass("active");
                    }
                });
            }
        }
    }


    Log.download = _ => {
        Utils.download(JSON.stringify(Log.data), `log_${Utils.timeFormat(Date.now(), true)}.json`, 'text/plain');
    }
    Log.upload = _ => {
        Utils.upload("application/json", files => {
            let file = files[0];
            let reader = new FileReader();
            reader.onload = e => {
                if (socket != null) {
                    if (socket.connected) {
                        socket.disconnect();
                    }
                }
                Log.setTitle(`[file] ${file.name}`);
                Log.reset();
                Log.update(JSON.parse(e.target.result));
            };
            reader.readAsText(file);
        });
    }
    Log.notificationState = notify => {
        return $notification_permission.hasClass("active") & $notification_states[notify].hasClass("active");
    }

    $(document).ready(_ => {
        $socket_switch = $("#log-socket-switch");
        $socket_switch.on("click", socketSwitch);
        $notification_permission = $("#log-notification-permission");
        if (Notification.permission == "granted") {
            $notification_permission.addClass("active");
        }
        $notification_permission.on("click", notificationPermission);
        $("a[role='log-tool-button'].log-notification-state").each(function () {
            let $this = $(this);
            let title = $this.attr("title");
            $notification_states[title] = $this;
            if (Log.notificationDefault[title]) {
                $this.addClass("active");
            }
            $this.on("click", _ => {
                notificationSwitch(title);
            });
        });
        if (typeof Log.socketURL !== "undefined") {
            socket = io(Log.socketURL, { autoConnect: false });
            socket.on("connect", _ => {
                $socket_switch.addClass("active");
                Log.reset();
                socket.emit("log_fetch", 0);
                Log.setTitle(`[socket] connected to ${Log.socketURL} (${socket.id})`)
            });
            socket.on("disconnect", _ => {
                $socket_switch.removeClass("active");
                Log.setTitle("[socket] disconnected");
            });
            socket.on("log_update", msgs => {
                Log.update(msgs);
                if (msgs[0].index == 0) {
                    Log.toBottom();
                }
            });
            socket.on("log_append", msg => {
                let end = msg.index;
                let start = Log.data.length;
                if (end > start) {
                    socket.emit("log_fetch", start, end);
                }
                Log.update(msg);
            });
            socket.connect();
        }
    });
})(window.Log = window.Log || {}, jQuery);

(function (Log, $) {
    const tolerance = 1;
    const titles = {
        success: "Success",
        warning: "Warning",
        error: "Error",
        info: "Info",
    }

    let notification = null;
    let $panel = null;
    let $title = null;
    let $jump_input = null;

    function jumpInputResize() {
        $jump_input.get(0).size = Math.max(1, $jump_input.val().length);
    }
    function renderEmpty(idx) {
        $panel.append(`<div role='log-item-empty' id='${idx}'></div>`);
    }
    function renderItem(msg, append = false) {
        let idx = msg.index;
        let time = Utils.timeFormat(msg.timestamp);
        let tex = msg.tex;
        let label = msg.label;
        let notify = label[0];
        let content = msg.message;

        let $labels = [];
        for (let l of label) {
            $labels.push(`<a role="log-label" class="padding-normal label" data-label="${l}" onclick="Log.filterSelect('${l}')"></a>`);
        }

        let $old = $panel.children(`#${idx}`);
        let doScroll = false;
        let selected = Log.filterSelected(idx);
        if (append & selected) {
            let $prev = $old.prevAll(":visible").first();
            if ($prev.length > 0) {
                doScroll = window.location.hash == `#${$prev.attr("id")}`;
            }
        }
        let $new = $(`
        <div role="log-item" id="${idx}" class="highlight">
            <div role="log-title">
                <div role="log-meta">
                    <a role="log-index" class="btn highlight padding-normal" href="#${idx}">${idx}</a>
                    <a role="log-timestamp" class="btn highlight padding-normal" href="#${idx}">${time}</a>
                </div>
                <div role="log-label-panel" class="padding-normal">
                    ${$labels.join("")}
                </div>
            </div>
            <div role="log-content" class="block highlight padding-normal">
                ${content}
            </div>
        </div>`);
        $old.replaceWith($new);
        if (tex) {
            renderMathInElement($new.get(0));
        }
        if (!selected) {
            $new.hide();
        }
        if (doScroll) {
            window.location.hash = `#${idx}`;
        }
        if (append & selected) {
            if (Log.notificationState(notify)) {
                if (notification != null) {
                    notification.close();
                }
                let kwargs = {
                    body: $new.children('div[role="log-content"]').text()
                }
                if (typeof Log.notificationIcons != "undefined") {
                    kwargs.icon = Log.notificationIcons[notify];
                }
                notification = new Notification(titles[notify], kwargs);
                notification.onclick = _ => {
                    window.location.hash = `#${idx}`;
                };
            }
        }
    }

    Log.data = [];
    Log.labels = {};
    Log.setTitle = title => {
        $title.text(title);
    }
    Log.reset = _ => {
        Log.data = [];
        $panel.empty();
    }
    Log.update = msgs => {
        let append = false;
        if (!(msgs instanceof Array)) {
            append = msgs.index >= Log.data.length;
            msgs = [msgs];
        }
        if (msgs.length > 0) {
            let end = msgs.at(-1).index;
            let start = Log.data.length;
            for (let i = start; i <= end; i++) {
                Log.data.push(null);
                renderEmpty(i);
            }
            for (let msg of msgs) {
                let idx = msg.index;
                if (Log.data[idx] == null) {
                    let labels = msg.label;
                    for (let l of labels) {
                        if (!(l in Log.labels)) {
                            Log.filterAddOption(l);
                            Log.labels[l] = new Set();
                        }
                        Log.labels[l].add(idx);
                    }
                    Log.data[idx] = msg;
                    renderItem(msg, append);
                }
            }
        }
    }
    Log.toTop = _ => {
        window.location.hash = "";
        let $first = $panel.children(":visible").first();
        if ($first.length > 0) {
            window.location.hash = `#${$first.attr("id")}`;
        }
    }
    Log.toBottom = _ => {
        window.location.hash = "";
        let $last = $panel.children(":visible").last();
        if ($last.length > 0) {
            window.location.hash = `#${$last.attr("id")}`;
        }
    }
    Log.jump = _ => {
        let val = $jump_input.val();
        if (val == "") {
            window.location.hash = "";
        }
        else {
            let idx = parseInt(val);
            if (idx >= 0 && idx < Log.data.length) {
                window.location.hash = `#${idx}`;
            }
        }
    }


    $(document).ready(_ => {
        $panel = $("#log-panel");
        $title = $("#log-title");
        $jump_input = $("#log-jump-input");
        $jump_input.on("input", jumpInputResize);
        $jump_input.on("keydown", e => {
            if (e.which == 13) {
                Log.jump();
            }
        });
        $jump_input.on("focus", _ => {
            window.location.hash = "";
        });
        jumpInputResize();
    });
})(window.Log = window.Log || {}, jQuery);

(function (Log, $) {
    let $filter_select = null;
    let $filter_logic = {};

    function filter() {
        $("div[role='log-item']").each(function () {
            let $this = $(this);
            let idx = parseInt($this.attr("id"));
            if (Log.filterSelected(idx)) {
                $this.show();
            }
            else {
                $this.hide();
            }
        });
    }
    function filterLogicSwitch(logic) {
        if (!$filter_logic[logic].hasClass("active")) {
            for (let [label, $btn] of Object.entries($filter_logic)) {
                if (label == logic) {
                    $btn.addClass("active");
                }
                else {
                    $btn.removeClass("active");
                }
            }
            filter();
        }
    }

    Log.filterSelected = idx => {
        let selected = $filter_select[0].selectize.items;
        if ($filter_logic.and.hasClass("active")) {
            for (let l of selected) {
                if (!Log.labels[l].has(idx)) {
                    return false;
                }
            }
            return true;
        }
        else if ($filter_logic.or.hasClass("active")) {
            for (let l of selected) {
                if (Log.labels[l].has(idx)) {
                    return true;
                }
            }
            return false;
        }
        else {
            return true;
        }
    }
    Log.filterSelect = label => {
        $filter_select[0].selectize.addItem(label);
    }
    Log.filterAddOption = label => {
        $filter_select[0].selectize.addOption({ label: label });
    }
    Log.filterClear = _ => {
        $filter_select[0].selectize.clear();
    }

    $(document).ready(_ => {
        $filter_select = $("#log-filter-select");
        $filter_select.selectize({
            labelField: "label",
            valueField: "label",
            searchField: "label",
            sortField: "label",
            plugins: ["remove_button"],
        });
        $filter_select[0].selectize.on("change", filter);
        $filter_select[0].selectize.on("focus", _ => {
            window.location.hash = "";
        });
        $("a[role='log-tool-button'].log-filter-logic").each(function () {
            let $this = $(this);
            let title = $this.attr("title");
            $filter_logic[title] = $this;
            $this.on("click", _ => {
                filterLogicSwitch(title);
            });
        });
        filterLogicSwitch("and");
    });
})(window.Log = window.Log || {}, jQuery);
