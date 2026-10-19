"""Links between accounts through shared devices and shipping addresses.

Accounts that have used the same phone or shipped to the same address are
joined into one connected component (union-find). A household is a component
of two to four people; a fraud ring is a component of a dozen-plus brand-new
accounts on two or three phones. Carrier IP addresses are deliberately not
used as links: thousands of honest strangers share them.
"""

from __future__ import annotations


class AccountGraph:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._members: dict[int, list[int]] = {}
        self.device_users: dict[int, set[int]] = {}
        self.address_users: dict[int, set[int]] = {}
        # The reverse index, for drawing an account's neighbourhood.
        self.account_devices: dict[int, set[int]] = {}
        self.account_addresses: dict[int, set[int]] = {}

    # ---------------------------------------------------------- union-find
    def _find(self, account: int) -> int:
        parent = self._parent.get(account)
        if parent is None:
            self._parent[account] = account
            self._members[account] = [account]
            return account
        root = account
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[account] != root:
            self._parent[account], account = root, self._parent[account]
        return root

    def _union(self, a: int, b: int) -> None:
        root_a, root_b = self._find(a), self._find(b)
        if root_a == root_b:
            return
        if len(self._members[root_a]) < len(self._members[root_b]):
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._members[root_a].extend(self._members.pop(root_b))

    # ------------------------------------------------------------------ api
    def link(self, account: int, device: int, address: int) -> None:
        self._find(account)
        if device >= 0:
            self.account_devices.setdefault(account, set()).add(device)
            users = self.device_users.setdefault(device, set())
            for other in users:
                self._union(account, other)
                break
            users.add(account)
        if address >= 0:
            self.account_addresses.setdefault(account, set()).add(address)
            users = self.address_users.setdefault(address, set())
            for other in users:
                self._union(account, other)
                break
            users.add(account)

    def device_accounts(self, device: int, exclude: int) -> int:
        users = self.device_users.get(device)
        return len(users) - (exclude in users) if users else 0

    def address_accounts(self, address: int, exclude: int) -> int:
        users = self.address_users.get(address)
        return len(users) - (exclude in users) if users else 0

    def component(self, account: int) -> list[int]:
        if account not in self._parent:
            return [account]
        return self._members[self._find(account)]

    def component_if_linked(self, account: int, device: int, address: int) -> list[int]:
        """The component `account` would belong to after this transaction, without changing anything."""
        roots = {self._find(account)} if account in self._parent else set()
        for users in (self.device_users.get(device) if device >= 0 else None, self.address_users.get(address) if address >= 0 else None):
            if users:
                roots.add(self._find(next(iter(users))))
        if not roots:
            return [account]
        members: list[int] = []
        for root in roots:
            members.extend(self._members[root])
        if account not in self._parent:
            members.append(account)
        return members

    def subgraph(self, account: int, limit: int = 80) -> dict:
        """Nodes and edges around an account's component, for drawing."""
        component = self.component(account)
        members = [account, *[member for member in component if member != account][: limit - 1]]
        member_set = set(members)
        devices = {device for member in members for device in self.account_devices.get(member, ())}
        addresses = {address for member in members for address in self.account_addresses.get(member, ())}
        edges = [("a", user, "d", device) for device in devices for user in self.device_users[device] if user in member_set]
        edges += [("a", user, "s", address) for address in addresses for user in self.address_users[address] if user in member_set]
        return {"accounts": members, "devices": sorted(devices), "addresses": sorted(addresses), "edges": edges}
